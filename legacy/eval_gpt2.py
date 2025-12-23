import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
from tqdm import tqdm

# ========= Configuration =========
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "/home/lingjie7/models/huggingface/openai-community/gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name).to(device).eval()


# ========= 1. Compute per-token log-likelihood =========
@torch.no_grad()
def log_likelihood(prompt: str, continuation: str) -> float:
    """
    Compute the *average* per-token log-likelihood of `continuation` given `prompt`.
    """
    full_text = prompt + continuation
    enc_full = tokenizer(full_text, return_tensors="pt").to(device)
    enc_prompt = tokenizer(prompt, return_tensors="pt").to(device)

    # Shift labels to ignore prompt tokens
    labels = enc_full.input_ids.clone()
    labels[:, : enc_prompt.input_ids.size(1)] = -100

    outputs = model(enc_full.input_ids, labels=labels)
    loss = outputs.loss.item()  # average NLL per token in continuation
    return -loss  # return mean log-prob (negative NLL)


import string
import torch
from tqdm import tqdm

def evaluate_lambada(dataset, limit=None, max_gen_tokens=10):
    """
    Evaluate GPT-2 on LAMBADA (last-word prediction).
    Generates up to `max_gen_tokens` tokens after the context,
    then compares the first decoded word with the gold target.
    Handles multi-token words like 'hastings'.
    """
    correct, total = 0, 0

    for sample in tqdm(dataset):
        text = sample["text"].strip()
        words = text.split()
        if len(words) < 2:
            continue

        context = " ".join(words[:-1])
        target = words[-1]

        # Tokenize input
        inputs = tokenizer(context, return_tensors="pt").to(device)

        # Generate continuation
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_gen_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

        # Decode generated continuation only (exclude context)
        gen_text = tokenizer.decode(outputs[0][inputs.input_ids.size(1):], skip_special_tokens=True)

        # Extract first predicted "word" from generation
        gen_first_word = gen_text.strip().split()[0] if gen_text.strip() else ""

        # Normalize both
        pred_norm = gen_first_word.strip(string.punctuation).lower()
        target_norm = target.strip(string.punctuation).lower()

        # Debug (optional)
        # print(f"CTX: {context[-80:]}")
        # print(f"PRED: '{pred_norm}' | GOLD: '{target_norm}'\n")

        if pred_norm == target_norm:
            correct += 1
        total += 1

        if limit and total >= limit:
            break

    acc = correct / total if total > 0 else 0
    print(f"[LAMBADA] Accuracy: {acc*100:.2f}% ({correct}/{total})")
    return acc




# ========= 3. Evaluate CBT (multiple-choice cloze) =========
def evaluate_cbt(dataset, limit=None):
    """
    Evaluate GPT-2 on CBT (CN/NE).
    Scores each candidate by its continuation likelihood
    and chooses the most probable one.
    """
    correct, total = 0, 0

    for sample in tqdm(dataset):
        context = " ".join(sample["sentences"])
        question = sample["question"]
        options = sample["options"]
        answer = sample["answer"]

        # Naturalistic prompt — story continuation, not QA format
        # The missing word 'XXXXX' is replaced by each candidate
        base_question = question.replace("XXXXX", "")
        base_prompt = context.strip() + "\n" + base_question.strip()

        # Score each candidate
        scores = {}
        for opt in options:
            # For consistent tokenization spacing
            cont = " " + opt.strip()
            scores[opt] = log_likelihood(base_prompt, cont)

        # Choose highest average log-likelihood
        pred = max(scores, key=scores.get)
        if pred.lower() == answer.lower():
            correct += 1
        total += 1

        if limit and total >= limit:
            break

    acc = correct / total if total > 0 else 0
    print(f"[CBT] Accuracy: {acc*100:.2f}% ({correct}/{total})")
    return acc


# ========= 4. Run evaluation =========
if __name__ == "__main__":
    # ---- LAMBADA ----
    lambada = load_dataset("/home/lingjie7/datasets/huggingface/lambada", split="test")
    print("\nEvaluating on LAMBADA...")
    evaluate_lambada(lambada, limit=200)
    breakpoint()
    # ---- CBT-CN ----
    cbt_cn = load_dataset("/home/lingjie7/datasets/huggingface/cam-cst/cbt", "CN", split="validation")
    print("\nEvaluating on CBT-CN...")
    evaluate_cbt(cbt_cn, limit=None)

    # ---- CBT-NE ----
    cbt_ne = load_dataset("/home/lingjie7/datasets/huggingface/cam-cst/cbt", "NE", split="validation")
    print("\nEvaluating on CBT-NE...")
    evaluate_cbt(cbt_ne, limit=None)
