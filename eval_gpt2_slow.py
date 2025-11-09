import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
from tqdm import tqdm

# ========== Configuration ==========
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "gpt2"  # can switch to "gpt2-medium" or larger
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name).to(device).eval()


# ========== 1. Evaluation helper ==========
@torch.no_grad()
def score_completion(prompt, target):
    """
    Compute the log-probability of the target as the next token(s) after the prompt.
    """
    full = prompt + target
    inputs = tokenizer(full, return_tensors="pt").to(device)
    with_target = inputs["input_ids"]
    # mask out prompt part
    target_ids = with_target.clone()
    target_ids[:, : tokenizer(prompt, return_tensors="pt").input_ids.shape[1]] = -100
    loss = model(with_target, labels=target_ids).loss
    return -loss.item()  # log-likelihood (negative loss)


def evaluate_cloze(dataset, prompt_field, target_field, is_last_word=False, limit=None):
    """
    Evaluate GPT-2 on a cloze-style dataset.
    - dataset: HuggingFace dataset object
    - prompt_field: str, field name for context/prompt
    - target_field: str, field name for answer
    - is_last_word: bool, whether the answer is at the end (like LAMBADA)
    - limit: optional limit for quick test
    """
    correct, total = 0, 0
    for sample in tqdm(dataset):
        prompt = sample[prompt_field].strip()
        target = sample[target_field].strip()

        # For LAMBADA, ensure prompt excludes target
        if is_last_word and prompt.endswith(target):
            prompt = prompt[: -len(target)].strip()

        # Generate next token
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False  # greedy prediction
        )
        pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        pred_next = pred_text[len(prompt):].strip().split()[0] if len(pred_text) > len(prompt) else ""

        # Compare first word of prediction to ground truth
        if pred_next.lower() == target.lower():
            correct += 1
        total += 1

        if limit and total >= limit:
            break

    acc = correct / total if total > 0 else 0.0
    print(f"Accuracy: {acc*100:.2f}% ({correct}/{total})")
    return acc


# ========== 2. Dataset loading & evaluation ==========

# ---- LAMBADA ----
lambada = load_dataset("lambada", "standard", split="test")
print("\nEvaluating on LAMBADA...")
evaluate_cloze(lambada, prompt_field="text", target_field="target", is_last_word=True, limit=200)

# ---- CBT-CN ----
cbt_cn = load_dataset("cbt", "CN", split="validation")
print("\nEvaluating on CBT-CN...")
evaluate_cloze(cbt_cn, prompt_field="context", target_field="answer", limit=200)

# ---- CBT-NE ----
cbt_ne = load_dataset("cbt", "NE", split="validation")
print("\nEvaluating on CBT-NE...")
evaluate_cloze(cbt_ne, prompt_field="context", target_field="answer", limit=200)
