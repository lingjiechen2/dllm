import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, BasicTokenizer
from datasets import load_dataset
from tqdm import tqdm
from sacremoses import MosesDetokenizer
import string

# ========== Configuration ==========
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "/mnt/lustrenew/mllm_aligned/shared/models/huggingface/openai-community/gpt2-large"  # or "gpt2-medium" / custom local path
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name).to(device).eval()
basic_tokenizer = BasicTokenizer(do_lower_case=False)
detokenizer = MosesDetokenizer(lang='en')


# ========== Preprocessing helpers ==========
def preprocess(text: str) -> str:
    """Normalize smart quotes and return trimmed text with a leading newline."""
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("''", '"').replace("``", '"')
    return '\n' + text.strip()


def detokenize(line: str) -> str:
    """Moses detokenizer to clean PTB artifacts."""
    toks = line.split()
    return detokenizer.detokenize(toks)


# ========== LM scoring ==========
@torch.no_grad()
def compute_logprob(prompt: str) -> float:
    """Compute the total log-likelihood (sum of log probs) for a given string."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    outputs = model(input_ids, labels=input_ids)
    neg_loss = -outputs.loss.item() * input_ids.shape[1]
    return neg_loss  # negative loss × sequence length = log-likelihood


def score_candidate(sentences, question, candidate) -> float:
    """Score the full text when the blank is filled with `candidate`."""
    context = "\n".join(sentences)
    filled = f"{context.strip()} {question.replace('XXXXX', candidate)}"
    breakpoint()
    filled = detokenize(preprocess(filled))
    return compute_logprob(filled)


# ========== CBT evaluation ==========
def evaluate_cbt(subset="CN", limit=None):
    """
    Evaluate GPT-2 on CBT (CN or NE subset).
    Each example has context (20 sentences), question (with 'XXXXX'), 10 candidates, and answer.
    """
    dataset = load_dataset("/mnt/lustrenew/mllm_aligned/shared/datasets/huggingface/cam-cst/cbt", subset, split="validation")

    correct, total = 0, 0
    for i, sample in enumerate(tqdm(dataset, desc=f"Evaluating CBT-{subset}")):
        sentences = sample["sentences"]
        question = sample["question"]
        candidates = sample["options"]
        answer = sample["answer"]

        # compute LM score for each candidate
        scores = [score_candidate(sentences, question, cand) for cand in candidates]
        pred = candidates[int(torch.tensor(scores).argmax())]

        if pred.lower() == answer.lower():
            correct += 1
        total += 1

        if limit and total >= limit:
            break

    acc = correct / total if total > 0 else 0.0
    print(f"[CBT-{subset}] Accuracy: {acc*100:.2f}% ({correct}/{total})")
    return acc


# ========== Run Evaluation ==========
if __name__ == "__main__":
    print("\nEvaluating GPT-2 on CBT (Common Nouns)...")
    evaluate_cbt("CN", limit=200)  # smaller subset for speed

    print("\nEvaluating GPT-2 on CBT (Named Entities)...")
    evaluate_cbt("NE", limit=200)
