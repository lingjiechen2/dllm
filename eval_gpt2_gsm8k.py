"""
python -u evaluate_gpt2_gsm8k.py --model_name_or_path "gpt2"
"""

from dataclasses import dataclass
import re
import transformers
import datasets
import torch


# ----------------------------
# Arguments
# ----------------------------
@dataclass
class ScriptArguments:
    model_name_or_path: str = "/mnt/lustrenew/mllm_aligned/shared/models/huggingface/openai-community/gpt2"
    max_new_tokens: int = 64
    max_samples: int = 30000     # evaluate subset for speed
    seed: int = 42
    device: str = "cuda"


parser = transformers.HfArgumentParser((ScriptArguments,))
script_args = parser.parse_args_into_dataclasses()[0]
transformers.set_seed(script_args.seed)
device = torch.device(script_args.device)


# ----------------------------
# Load model + tokenizer
# ----------------------------
tokenizer = transformers.AutoTokenizer.from_pretrained(script_args.model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token

model = transformers.AutoModelForCausalLM.from_pretrained(
    script_args.model_name_or_path
).to(device).eval()
model.config.pad_token_id = tokenizer.eos_token_id


# ----------------------------
# Helper: extract final number
# ----------------------------
def extract_number(s):
    nums = re.findall(r"-?\d+", s)
    return int(nums[-1]) if nums else None


# ----------------------------
# Generation: step-by-step greedy
# ----------------------------
def generate_answer(prompt, max_steps):
    text = prompt
    for _ in range(max_steps):
        inputs = tokenizer(text, return_tensors="pt").to(device)
        out = model.generate(**inputs, max_new_tokens=1, do_sample=False)
        decoded = tokenizer.decode(out[0], skip_special_tokens=True)

        # get only the newly added token
        new_part = decoded[len(text):]
        if not new_part:
            break

        text += new_part
        if "\n" in new_part or "." in new_part:
            # heuristic: stop on newline or sentence break
            break

    return text


# ----------------------------
# Load GSM8K
# ----------------------------
gsm8k = datasets.load_dataset("gsm8k", "main")["test"]
gsm8k = gsm8k.select(range(min(script_args.max_samples, len(gsm8k))))


# ----------------------------
# Evaluation loop
# ----------------------------
import json
from tqdm import tqdm

correct = 0
log_path = "results_gsm8k_gpt2.jsonl"
log_file = open(log_path, "w")

pbar = tqdm(range(len(gsm8k)), total=len(gsm8k), desc="Evaluating GPT-2")

for idx in pbar:
    ex = gsm8k[idx]

    question = ex["question"]
    gt = extract_number(ex["answer"])
    prompt = f"Q: {question}\nA:"

    # GPT-2 step-by-step greedy generation
    full_output = generate_answer(prompt, script_args.max_new_tokens)
    pred = extract_number(full_output)

    # --- Logging ---
    record = {
        "idx": idx,
        "question": question,
        "gt_answer": gt,
        "pred_answer": pred,
        "generated_text": full_output,
    }
    log_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    # --- Accuracy update ---
    if pred == gt:
        correct += 1

    running_acc = correct / (idx + 1)
    pbar.set_postfix({"acc": f"{running_acc:.4f}"})

log_file.close()

final_acc = correct / len(gsm8k)
print("\n" + "=" * 60)
print(f"Final GSM8K Accuracy ({script_args.model_name_or_path}): {final_acc:.4f}")
print("=" * 60)
