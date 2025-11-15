import os, math, json
from dataclasses import dataclass

import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset
import transformers

# ------------------------- Label Computation ------------------------- #

def compute_labels_ar(input_ids: torch.Tensor):
    # input_ids: (T,)
    T = input_ids.size(0)
    L = T
    pos = torch.arange(T)

    labels = {
        "L": torch.full((T,), L, dtype=torch.float32),
        "R": (L - 1 - pos).float(),

        # AR models have no mask structure → fill zeros
        "B": torch.zeros(T, dtype=torch.float32),
        "B_all": torch.zeros(T, dtype=torch.float32),
    }
    return labels


# ---------------------- AR Hidden Extraction ------------------------ #

@torch.no_grad()
def generate_labelled_data_ar(
    model,
    tokenizer,
    inputs,
    probe_layers,
):
    device = next(model.parameters()).device

    if isinstance(inputs[0], list):
        ids = torch.as_tensor(inputs[0], dtype=torch.long, device=device)
    else:
        ids = inputs[0].to(device)
    breakpoint()
    out = model(
        ids.unsqueeze(0),
        output_hidden_states=True
    )

    hidden_states = out.hidden_states
    T = ids.size(0)

    labels = compute_labels_ar(ids)

    flat_hidden = {}
    for lid in probe_layers:
        h = hidden_states[lid][0]           # (T, D)
        flat_hidden[lid] = h.cpu()

    probe_data = {
        "hidden_states": flat_hidden,
        "labels": {k: v.cpu() for k, v in labels.items()},
        "steps": 1,                # fixed for AR
        "max_new_tokens": T,
        "eff_steps": [0],          # AR = single snapshot
    }

    return ids.unsqueeze(0), probe_data


# ---------------------- Saving ---------------------- #

def save_probe_data_grouped(probe_data, model_tag, dataset_tag, split_idx):
    steps = probe_data["steps"]
    L_len = probe_data["max_new_tokens"]
    labels = probe_data["labels"]
    eff_steps = probe_data["eff_steps"]

    for layer_id, h in probe_data["hidden_states"].items():
        layer_dir = f"probe_data/{model_tag}/{dataset_tag}/layer{layer_id:02d}"
        os.makedirs(layer_dir, exist_ok=True)

        save_path = os.path.join(
            layer_dir,
            f"L{L_len}_step{steps}_split{split_idx:02d}.pt"
        )

        torch.save(
            {
                "hidden_states": h.to(torch.float32).numpy(),
                "B": labels["B"].numpy(),
                "B_all": labels["B_all"].numpy(),
                "R": labels["R"].numpy(),
                "L": labels["L"].numpy(),
                "steps": steps,
                "L_len": L_len,
                "eff_steps": eff_steps,
                "split_idx": split_idx,
            },
            save_path,
        )

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


# ---------------------- Main ---------------------- #

if __name__ == "__main__":
    import dllm  # Only needed if your tokenizer/model come from dllm ecosystem

    @dataclass
    class ScriptArguments:
        model_name_or_path: str = "/mnt/lustrenew/mllm_aligned/shared/models/huggingface/meta-llama/Meta-Llama-3-8B-Instruct/"  # example AR model
        dataset_name: str = "openai/gsm8k"
        dataset_config: str = "main"
        dataset_split: str = "test"
        model_tag: str = "ar_baseline"
        seed: int = 42
        split_idx: int = 0
        num_splits: int = 1

    parser = transformers.HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    transformers.set_seed(script_args.seed)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        torch_dtype=torch.float16,
        device_map="auto"
    ).eval()

    tokenizer = transformers.AutoTokenizer.from_pretrained(script_args.model_name_or_path)

    if script_args.dataset_config:
        dataset = load_dataset(script_args.dataset_name, script_args.dataset_config)[script_args.dataset_split]
    else:
        dataset = load_dataset(script_args.dataset_name)[script_args.dataset_split]

    probe_layers = [1, 8, 16, 24]

    total = len(dataset)
    chunk = math.ceil(total / script_args.num_splits)
    start = script_args.split_idx * chunk
    end = min((script_args.split_idx + 1) * chunk, total)

    print(f"[Split] {start} → {end} / {total}  (split {script_args.split_idx}/{script_args.num_splits})")

    subset = dataset.select(range(start, end))

    for idx, sample in enumerate(tqdm(subset, dynamic_ncols=True)):
        q = sample["question"]
        msg = [{"role": "user", "content": q}]
        ids = tokenizer.apply_chat_template(msg, add_generation_prompt=True, tokenize=True)

        _, probe_data = generate_labelled_data_ar(
            model=model,
            tokenizer=tokenizer,
            inputs=[ids],
            probe_layers=probe_layers,
        )

        save_probe_data_grouped(
            probe_data,
            model_tag=script_args.model_tag,
            dataset_tag=script_args.dataset_name.split("/")[-1],
            split_idx=script_args.split_idx,
        )

    print("[Done] AR probe extraction complete.")
