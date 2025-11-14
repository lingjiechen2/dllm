import math, os, json

from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dllm.utils.generation_utils import get_num_transfer_tokens
from dllm.pipelines.llada.generator import LLaDAGeneratorConfig
from dllm.core.generation.generator import GeneratorOutput


# ------------------------- Helper Functions ------------------------- #

def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Add Gumbel noise for exploration (correct formulation)."""
    if temperature == 0:
        return logits
    noise = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(noise + 1e-9))
    return logits + gumbel_noise * temperature


def compute_labels(x: torch.Tensor, attention_mask: torch.Tensor, mask_id: int):
    """
    Compute:
      - B_i:  # of mask tokens to the right of each token (local)
      - B_all: total # of mask tokens in the entire valid sequence (global)
      - R_i:  distance to end (L - i)
      - L:    sequence length per token
    """
    B, T = x.shape
    B_i = torch.zeros_like(x, dtype=torch.float)
    B_all = torch.zeros_like(x, dtype=torch.float)
    R_i = torch.zeros_like(x, dtype=torch.float)
    L_vals = torch.zeros_like(x, dtype=torch.float)

    for b in range(B):
        valid_idx = attention_mask[b].bool()
        seq = x[b, valid_idx]
        L_val = seq.numel()

        # total mask count within the sequence
        total_masks = (seq == mask_id).sum().item()

        for i in range(L_val):
            B_i[b, i] = (seq[i + 1 :] == mask_id).sum().item()
            B_all[b, i] = total_masks
            R_i[b, i] = L_val - i
            L_vals[b, i] = L_val

    return {"B": B_i, "B_all": B_all, "R": R_i, "L": L_vals}


def choose_extract_steps(effective_steps: int, num_extract: int) -> list[int]:
    """
    Sample num_extract timestamps uniformly from [1, effective_steps-1].
    No manual final-step enforcement.
    """
    if num_extract <= 0:
        return []

    # avoid step 0 because that is trivial prefill
    candidates = list(range(1, effective_steps))
    num_extract = min(num_extract, len(candidates))

    steps = np.random.choice(candidates, size=num_extract, replace=False)
    return sorted(int(s) for s in steps)


# ---------------------- Core Generation & Probing -------------------- #

@torch.no_grad()
def generate_labelled_data(
    generator_instance,
    inputs: list[torch.Tensor | list[int]],
    config: LLaDAGeneratorConfig | None = None,
    probe_layers: list[int] | None = None,
    **kwargs,
) -> tuple[GeneratorOutput, dict[str, torch.Tensor]]:
    """Run DLLM generation and return collected hidden states + labels."""
    if config is None:
        config = LLaDAGeneratorConfig()

    steps = kwargs.get("steps", config.steps)
    max_new_tokens = kwargs.get("max_new_tokens", config.max_new_tokens)
    temperature = kwargs.get("temperature", config.temperature)
    cfg_scale = kwargs.get("cfg_scale", config.cfg_scale)
    cfg_keep_tokens = kwargs.get("cfg_keep_tokens", config.cfg_keep_tokens)
    remasking = kwargs.get("remasking", config.remasking)
    stochastic_transfer = kwargs.get("stochastic_transfer", config.stochastic_transfer)
    return_dict_in_generate = kwargs.get("return_dict_in_generate", config.return_dict_in_generate)
    confidence_eos_eot_inf = kwargs.get("confidence_eos_eot_inf", config.confidence_eos_eot_inf)
    num_extract_steps = kwargs.get("num_extract_steps", getattr(config, "num_extract_steps", 1))

    mask_id = generator_instance.tokenizer.mask_token_id
    eos_id = generator_instance.tokenizer.eos_token_id

    if isinstance(inputs[0], list):
        inputs = [torch.as_tensor(p, dtype=torch.long, device=generator_instance.model.device) for p in inputs]
    prompt_lens = [p.shape[0] for p in inputs]
    max_prompt_len = max(prompt_lens)
    max_length = max_new_tokens + max_prompt_len
    B, T = len(inputs), max_length

    x = torch.full((B, T), eos_id, dtype=torch.long, device=generator_instance.model.device)
    for i, p in enumerate(inputs):
        x[i, : prompt_lens[i]] = p
        x[i, prompt_lens[i] : prompt_lens[i] + max_new_tokens] = mask_id
    attention_mask = (x != eos_id).long()

    unmasked_index = (x != mask_id) & (x != eos_id)
    if cfg_keep_tokens:
        keep_mask = torch.isin(x, torch.as_tensor(cfg_keep_tokens, device=generator_instance.model.device))
        unmasked_index &= ~keep_mask

    histories = [x.clone()] if return_dict_in_generate else None
    probe_layers = probe_layers or [1, 8, 16, 24]
    
    flat_hidden = {lid: [] for lid in probe_layers}
    flat_labels = {k: [] for k in ["B", "B_all", "R", "L"]}

    # ----- diffusion steps -----
    num_transfer_tokens = get_num_transfer_tokens(
        mask_index=(x == mask_id),
        steps=steps,
        scheduler=generator_instance.scheduler,
        stochastic=stochastic_transfer,
    )
    effective_steps = num_transfer_tokens.size(1)

    extract_steps = choose_extract_steps(effective_steps, num_extract_steps)
    probe_steps: list[int] = []

    for step_idx in range(effective_steps):
        mask_index = x == mask_id

        if cfg_scale > 0.0:
            un_x = x.clone()
            un_x[unmasked_index] = mask_id
            x_ = torch.cat([x, un_x], dim=0)
            output = generator_instance.model(x_, attention_mask=attention_mask, output_hidden_states=True)
            logits, hidden_states = output.logits, output.hidden_states
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            hidden_states = tuple(h[:B] for h in hidden_states)
        else:
            output = generator_instance.model(x, attention_mask=attention_mask, output_hidden_states=True)
            logits, hidden_states = output.logits, output.hidden_states

        logits_with_noise = add_gumbel_noise(logits, temperature)
        x0 = torch.argmax(logits_with_noise, dim=-1)
        if confidence_eos_eot_inf:
            logits_with_noise[:, :, 126081] = -torch.inf
            logits_with_noise[:, :, 126348] = -torch.inf

        # compute confidence
        if remasking == "low_confidence":
            p = F.softmax(logits, dim=-1)
            x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
        elif remasking == "random":
            x0_p = torch.rand_like(x0, dtype=torch.float)
        else:
            raise NotImplementedError(remasking)

        x0 = torch.where(mask_index, x0, x)
        confidence = torch.where(mask_index, x0_p, -np.inf)

        transfer_index = torch.zeros_like(x0, dtype=torch.bool)
        for j in range(confidence.size(0)):
            _, idx = torch.topk(confidence[j], k=num_transfer_tokens[j, step_idx])
            transfer_index[j, idx] = True

        # ---- snapshot at selected timesteps ----
        
        if step_idx in extract_steps:
            labels = compute_labels(x, attention_mask, mask_id)
            assert x.size(0) == 1

            # flatten labels
            for key in flat_labels:
                flat_labels[key].append(labels[key].view(-1).detach().cpu())

            # flatten hidden states
            for layer_id in probe_layers:
                h = hidden_states[layer_id]      # (1, T, D)
                T, D = h.size(1), h.size(2)
                flat_hidden[layer_id].append(h.view(T, D).detach().cpu())

            probe_steps.append(step_idx)
            if len(probe_steps) == len(extract_steps):
                break

        # update x for next diffusion step
        x[transfer_index] = x0[transfer_index]
        if histories is not None:
            histories.append(x.clone())

    if not probe_steps:
        raise RuntimeError("No extraction steps were collected. Check num_extract_steps / extract_steps logic.")

    # ---- concatenate flattened outputs ----
    labels_agg = {
        key: torch.cat(tensors, dim=0) for key, tensors in flat_labels.items()
    }  # each: (N_tokens,)
    hidden_agg = {
        lid: torch.cat(tensors, dim=0) for lid, tensors in flat_hidden.items()
    }  # each: (N_tokens, D)

    probe_data = {
        "hidden_states": hidden_agg,
        "labels": labels_agg,
        "steps": steps,
        "max_new_tokens": max_new_tokens,
        "eff_steps": probe_steps,  # list[int]
    }

    output = GeneratorOutput(sequences=x, histories=histories) if return_dict_in_generate else x
    return output, probe_data


# ---------------------- Saving Based on Hierarchy -------------------- #
def save_probe_data_grouped(probe_data, model_tag, dataset_tag, split_idx):
    steps = probe_data["steps"]
    L_len = probe_data["max_new_tokens"]
    labels = probe_data["labels"]
    eff_steps = probe_data.get("eff_steps", None)

    for layer_id, h in probe_data["hidden_states"].items():

        # NEW DIRECTORY STRUCTURE (no step folder)
        layer_dir = f"/mnt/lustrenew/mllm_safety-shared/fanyuyu/probe_data/{model_tag}/{dataset_tag}/layer{layer_id:02d}"
        os.makedirs(layer_dir, exist_ok=True)

        # NEW FILE NAME FORMAT
        save_path = os.path.join(
            layer_dir,
            f"L{L_len}_step{steps}_split{split_idx:02d}.pt"
        )

        torch.save(
            {
                "hidden_states": h.to(torch.float32).cpu().numpy(),
                "B": labels["B"].cpu().numpy(),
                "B_all": labels["B_all"].cpu().numpy(),
                "R": labels["R"].cpu().numpy(),
                "L": labels["L"].cpu().numpy(),
                "steps": steps,
                "L_len": L_len,
                "eff_steps": eff_steps,
                "split_idx": split_idx,
            },
            save_path,
        )

        # NEW META INDEX FILE (still appended)
        meta_path = os.path.join(os.path.dirname(layer_dir), "meta_index.jsonl")
        entry = {
            "layer": layer_id,
            "steps": steps,
            "L": L_len,
            "eff_steps": eff_steps,
            "split": split_idx,
            "path": save_path
        }
        with open(meta_path, "a") as f:
            f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    import transformers
    from dataclasses import dataclass
    from datasets import load_dataset
    import dllm
    from dllm.pipelines import llada

    # ---------------------- Argument definitions ---------------------- #
    @dataclass
    class ScriptArguments:
        model_name_or_path: str = "GSAI-ML/LLaDA-8B-Instruct"
        dataset_name: str = "openai/gsm8k"
        dataset_config: str = "main"
        dataset_split: str = "test"
        model_tag: str = "llada_instruct"
        seed: int = 42
        dtype: str = "float32"
        split_idx: int = 0
        num_splits: int = 1

        def __post_init__(self):
            self.model_name_or_path = dllm.utils.resolve_with_base_env(
                self.model_name_or_path, "BASE_MODELS_DIR"
            )

    @dataclass
    class GeneratorConfig(llada.LLaDAGeneratorConfig):
        steps: int = 1024
        max_new_tokens: int = 1024
        temperature: float = 0.0
        remasking: str = "low_confidence"
        num_extract_steps: int = 4   # collect 4 diffusion timesteps

    # ----------------------- Initialization ----------------------- #
    parser = transformers.HfArgumentParser((ScriptArguments, GeneratorConfig))
    script_args, gen_config = parser.parse_args_into_dataclasses()
    transformers.set_seed(script_args.seed)

    print("\n[Setup] Loading model and tokenizer...")
    model = dllm.utils.get_model(model_args=script_args).eval()
    tokenizer = dllm.utils.get_tokenizer(model_args=script_args)
    generator = llada.LLaDAGenerator(model=model, tokenizer=tokenizer)

    # ----------------------- Dataset load -------------------------- #
    if script_args.dataset_config:
        dataset = load_dataset(
            script_args.dataset_name, script_args.dataset_config
        )[script_args.dataset_split]
    else:
        dataset = load_dataset(script_args.dataset_name)[script_args.dataset_split]

    probe_layers = [1, 8, 16, 24]

    # ---- determine split ----
    total = len(dataset)
    chunk = math.ceil(total / script_args.num_splits)
    start = script_args.split_idx * chunk
    end   = min((script_args.split_idx + 1) * chunk, total)

    print(f"[Split] Processing samples {start} → {end} (split {script_args.split_idx}/{script_args.num_splits})")
    subset = dataset.select(range(start, end))

    # ============================================================
    #  MAIN EXTRACTION LOOP (AGGREGATED)
    # ============================================================
    agg_hidden = {lid: [] for lid in probe_layers}
    agg_labels = {k: [] for k in ["B", "B_all", "R", "L"]}
    agg_eff_steps = []

    for idx, sample in enumerate(tqdm(subset, desc="Extracting probe data", dynamic_ncols=True)):
        question = sample["question"]
        msg = [{"role": "user", "content": question}]
        inputs = [tokenizer.apply_chat_template(msg, add_generation_prompt=True, tokenize=True)]

        output, probe_data = generate_labelled_data(
            generator_instance=generator,
            inputs=inputs,
            config=gen_config,
            probe_layers=probe_layers,
            return_dict_in_generate=True,
        )
        # --------------------------
        # ACCUMULATE (NOT SAVE YET)
        # --------------------------
        for key in agg_labels:
            agg_labels[key].append(probe_data["labels"][key])  # shape: (N_tokens_sample,)

        for lid in probe_layers:
            agg_hidden[lid].append(probe_data["hidden_states"][lid])  # shape: (N_tokens_sample, D)

        agg_eff_steps.extend(probe_data["eff_steps"])

        if (idx + 1) % 50 == 0:
            decoded = tokenizer.decode(output.sequences[0])
            tqdm.write(f"[{idx+1}] {decoded[:200]} ...")

    # ============================================================
    #  CONCATENATE ALL SAMPLES & SAVE
    # ============================================================

    print("\n[Info] Concatenating and saving aggregated results...")

    probe_data_final = {
        "hidden_states": {
            lid: torch.cat(agg_hidden[lid], dim=0)
            for lid in probe_layers
        },
        "labels": {
            k: torch.cat(agg_labels[k], dim=0)
            for k in agg_labels
        },
        "steps": gen_config.steps,
        "max_new_tokens": gen_config.max_new_tokens,
        "eff_steps": agg_eff_steps,
        "num_samples": len(subset),
    }
    save_probe_data_grouped(
        probe_data_final,
        model_tag=script_args.model_tag,
        dataset_tag=script_args.dataset_name.split("/")[-1],
        split_idx=script_args.split_idx,
    )

    print("[Done] Probe data extraction complete.")