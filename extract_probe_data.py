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
    collected = {lid: [] for lid in probe_layers}

    # ----- diffusion steps -----
    num_transfer_tokens = get_num_transfer_tokens(
        mask_index=(x == mask_id),
        steps=steps,
        scheduler=generator_instance.scheduler,
        stochastic=stochastic_transfer,
    )
    effective_steps = num_transfer_tokens.size(1)

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

        if step_idx == effective_steps - 10:
            for layer_id in probe_layers:
                collected[layer_id].append(hidden_states[layer_id].detach().cpu())
            labels = compute_labels(x, attention_mask, mask_id)
            break
        
        x[transfer_index] = x0[transfer_index]
        if histories is not None:
            histories.append(x.clone())


    # concatenate all collected batches
    probe_data = {
        "hidden_states": {lid: torch.cat(collected[lid], dim=0) for lid in probe_layers},
        "labels": labels,
        "steps": steps,
        "max_new_tokens": max_new_tokens,
    }
    output = GeneratorOutput(sequences=x, histories=histories) if return_dict_in_generate else x
    return output, probe_data


# ---------------------- Saving Based on Hierarchy -------------------- #

def save_probe_data_grouped(probe_data, model_tag, dataset_tag):
    """Save one generation-config's results into layer-based .pt files."""
    steps = probe_data["steps"]
    L_len = probe_data["max_new_tokens"]
    labels = probe_data["labels"]
    num_samples = probe_data.get("num_samples", None)
    deduped_L = probe_data.get("deduped_L", None)

    for layer_id, h in probe_data["hidden_states"].items():
        layer_dir = f"probe_data/{model_tag}/{dataset_tag}/layer{layer_id:02d}/step{steps:03d}"
        os.makedirs(layer_dir, exist_ok=True)
        save_path = os.path.join(layer_dir, f"L{L_len}.pt")

        # save tensor data
        torch.save(
            {
                "hidden_states": h.numpy(),                      # (N_tokens, d)
                "B":          labels["B"].cpu().numpy(),         # (N_tokens,)
                "B_all":      labels["B_all"].cpu().numpy(),     # (N_tokens,)
                "R":          labels["R"].cpu().numpy(),         # (N_tokens,)
                "L":          labels["L"].cpu().numpy(),         # (N_tokens,)
            },
            save_path,
        )

        # write to meta index
        meta_path = os.path.join(os.path.dirname(layer_dir), "meta_index.jsonl")
        entry = {"layer": layer_id, "step": steps, "L": L_len, "path": save_path}
        with open(meta_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

        # write summary metadata
        summary_path = os.path.join(os.path.dirname(layer_dir), "meta_summary.json")

        summary_entry = {
            "layer": layer_id,
            "steps": steps,
            "L": L_len,
        }
        if num_samples is not None:
            summary_entry["num_samples"] = num_samples
        if deduped_L is not None:
            summary_entry["deduped_L"] = deduped_L

        # append entry to JSON list
        if os.path.exists(summary_path):
            with open(summary_path, "r") as f:
                try:
                    existing = json.load(f)
                    if not isinstance(existing, list):
                        existing = [existing]
                except json.JSONDecodeError:
                    existing = []
        else:
            existing = []

        existing.append(summary_entry)

        with open(summary_path, "w") as f:
            json.dump(existing, f, indent=2)

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
        seed: int = 42
        visualize: bool = True
        dtype: str = "float32"

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

    start = 0
    end = 300

    # ----------------------- Initialization ----------------------- #
    parser = transformers.HfArgumentParser((ScriptArguments, GeneratorConfig))
    script_args, gen_config = parser.parse_args_into_dataclasses()
    transformers.set_seed(script_args.seed)

    print("\n[Setup] Loading model and tokenizer...")
    model = dllm.utils.get_model(model_args=script_args).eval()
    tokenizer = dllm.utils.get_tokenizer(model_args=script_args)
    generator = llada.LLaDAGenerator(model=model, tokenizer=tokenizer)

    # ----------------------- Dataset load -------------------------- #
    dataset = load_dataset("openai/gsm8k", "main")["train"]
    probe_layers = [1, 8, 16, 24]
    collected_batches = {lid: [] for lid in probe_layers}
    collected_labels = []  # store labels per sample

    print(f"[Info] Starting probe extraction for {len(dataset)} GSM8K samples...")
    subset = dataset.select(range(start, end))
    for idx, sample in enumerate(tqdm(subset, desc="Extracting probe data", dynamic_ncols=True)):
        question = sample["question"]
        message = [{"role": "user", "content": question}]
        inputs = [tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=True)]

        output, probe_data = generate_labelled_data(
            generator_instance=generator,
            inputs=inputs,
            config=gen_config,
            probe_layers=probe_layers,
            return_dict_in_generate=True,
        )

        # (Optional) decode occasionally to avoid slowdown
        if (idx + 1) % 50 == 0:
            decoded = tokenizer.decode(output.sequences[0])
            tqdm.write(f"[{idx+1}] {decoded[:200]}...")  # use tqdm.write instead of print

        # accumulate results
        for layer_id in probe_layers:
            collected_batches[layer_id].append(probe_data["hidden_states"][layer_id].cpu())
        collected_labels.append(probe_data["labels"])

    # ----------------------- Save grouped results ----------------------- #
    print("[Info] Flattening and saving probe data...")

    # Initialize result container
    probe_data_final = {
        "hidden_states": {},   # per-layer flattened embeddings
        "labels": {},          # global flattened labels
    }

    # -------------------------------------------------------------------
    # 1️⃣  Flatten labels ONCE (shared across all layers)
    # -------------------------------------------------------------------
    ref_layer = probe_layers[0]
    labels_flat = {"B": [], "B_all": [], "R": [], "L": []}
    sample_lengths = []  # store each sample's seq_len

    for h_ref, lbl in zip(collected_batches[ref_layer], collected_labels):
        seq_len = h_ref.shape[1]
        lbl_cpu = {k: v[0, :seq_len].cpu() for k, v in lbl.items()}
        valid_mask = lbl_cpu["L"] > 0
        sample_lengths.append(int(valid_mask.sum()))  # record each sample's true length

        for key in ["B", "B_all", "R", "L"]:
            labels_flat[key].append(lbl_cpu[key][valid_mask])

    # Concatenate all label segments across samples
    for key in labels_flat:
        labels_flat[key] = torch.cat(labels_flat[key], dim=0)

    probe_data_final["labels"] = labels_flat
    probe_data_final["num_samples"] = len(sample_lengths)
    probe_data_final["deduped_L"] = sample_lengths


    # -------------------------------------------------------------------
    # 2️⃣  Flatten hidden states PER LAYER (aligned to label token order)
    # -------------------------------------------------------------------
    for layer_id in probe_layers:
        flat_h_list = []
        for h, lbl in zip(collected_batches[layer_id], collected_labels):
            seq_len = h.shape[1]
            valid_mask = lbl["L"][0, :seq_len].cpu() > 0
            flat_h_list.append(h[0, valid_mask, :])
        probe_data_final["hidden_states"][layer_id] = torch.cat(flat_h_list, dim=0)

    # -------------------------------------------------------------------
    # 3️⃣  Save metadata for reproducibility
    # -------------------------------------------------------------------
    probe_data_final["steps"] = gen_config.steps
    probe_data_final["max_new_tokens"] = gen_config.max_new_tokens
    save_probe_data_grouped(
        probe_data_final,
        model_tag="llada_instruct",
        dataset_tag="gsm8k",
    )
    print("[Done] Probe data extraction complete and saved.")
