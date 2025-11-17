import json
import os
from typing import List, Dict
import datasets
from tqdm import tqdm

###############################################
# 1. Teacher model wrapper
###############################################

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "/mnt/lustrenew/mllm_aligned/shared/models/huggingface/Qwen/Qwen3-32B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
model.to(DEVICE)
model.eval()


def call_llm(prompt: str, max_new_tokens: int = 512, temperature: float = 0.2) -> str:
    """Call your HF causal LLM and return the decoded string."""
    # 1. Wrap into chat format (system + user message)
    messages = [
        {"role": "system", "content": "You are a helpful and concise assistant."},
        {"role": "user", "content": prompt},
    ]
    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,  # adds the <|assistant|> tag at the end
        enable_thinking=False
    )

    # 2. Tokenize and move to device
    inputs = tokenizer(chat_text, return_tensors="pt").to(DEVICE)

    # 3. Generate
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # 4. Decode and slice off the prompt part
    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    # print(text)
    # breakpoint()

    return text

###############################################
# 2. Templates for each granularity
###############################################

GRANULARITY_SPECS = {
    32:  {"name": "MVA",        "instruction": "Provide ONLY `Ans: <value>` in ≤32 tokens."},
    64:  {"name": "ANS+WHY",    "instruction": "Output `<short clause>` followed by `Ans: <value>` at the very end (≤64 tokens)."},
    128: {"name": "BULLETS3",   "instruction": "Give exactly 3 bullets, each ≤20 tokens, ending with `Ans: <value>`."},
    256: {"name": "SHORT_COT",  "instruction": "Give a short reasoning ≤8 steps, then `Ans: <value>`."},
    512: {"name": "COT+VERIFY", "instruction": "Give detailed reasoning, 1-line verification, then `Ans: <value>`."},
    1024:{"name": "FULL_COT",   "instruction": "Give full reasoning/explanation then end with `Ans: <value>`."}
}


###############################################
# 3. Build compression prompt (no facts)
###############################################

def build_compression_prompt(question: str, answer: str, L: int) -> str:
    spec = GRANULARITY_SPECS[L]

    # special guidance for tiny budgets
    small_budget_hint = ""
    if L <= 32:
        small_budget_hint = (
            "- For this very small budget, provide **only** the final answer line.\n"
            "- Do not include any reasoning, explanation, or extra words.\n"
        )
    
    # special guidance for ANS+WHY format
    ans_why_hint = ""
    if L == 64:
        ans_why_hint = (
            "- Start with '<brief explanation>', then end with 'Ans: <value>'.\n"
            "- Do NOT repeat 'Ans:' multiple times. It should appear ONLY ONCE at the very end.\n"
        )

    return f"""
You are generating an answer under a strict token budget.

QUESTION:
{question}

CORRECT FINAL ANSWER:
{answer}

TARGET FORMAT ({spec['name']}):
{spec['instruction']}

RULES:
- Must be ≤ {L} tokens.
{small_budget_hint}{ans_why_hint}- If reasoning or explanation is allowed, it should come **before** the final answer.
- The final answer must appear exactly once, **at the end**, in the form `Ans: <value>`.
- No apologies, no meta text, no repetition.
- Avoid irrelevant or decorative content.

Now produce only the {spec['name']} output, ending with the final answer line.
""".strip()


###############################################
# 4. Generate all budgets for one QA pair
###############################################
eval_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def evaluate_generation(text: str, answer: str, L: int) -> dict:
    """Evaluate one generated answer with three simple metrics."""
    # 1. token length & budget
    tokens = eval_tokenizer.encode(text, add_special_tokens=False)
    token_len = len(tokens)
    within_budget = token_len <= L

    # 2. final-answer containment
    contains_final_answer = answer.strip().lower() in text.lower()

    # 3. correctness (parse out what follows after "Ans:")
    extracted = ""
    if "ans:" in text.lower():
        extracted = text.lower().split("ans:")[-1].strip().split()[0]  # first token after Ans:
    correctness = extracted == answer.strip().lower()

    return {
        "token_len": token_len,
        "within_budget": within_budget,
        "contains_final_answer": contains_final_answer,
        "correctness": correctness,
    }

def generate_multi_granularity(question: str, answer: str) -> List[Dict]:
    results = []
    for L in GRANULARITY_SPECS.keys():
        prompt = build_compression_prompt(question, answer, L)
        text = call_llm(prompt, max_new_tokens = L)
        metrics = evaluate_generation(text, answer, L)
        results.append({
            "L": L,
            "format": GRANULARITY_SPECS[L]["name"],
            "text": text
        })
    return results


###############################################
# 5. Build dataset entry
###############################################

def build_entry(qid: str, question: str, answer: str) -> dict:
    budgets = generate_multi_granularity(question, answer)
    return {
        "id": qid,
        "question": question,
        "answer_key": answer,
        "budgets": budgets,
    }

###############################################
# 6. Main builder loop (Dataset version)
###############################################

def build_dataset_from_hf(input_dataset: datasets.Dataset, output_path: str):
    """
    Input: HuggingFace Dataset with columns ['question', 'answer']
    """
    # os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as fout:
        for i, rec in enumerate(tqdm(input_dataset, desc="Processing questions")):
            qid = rec.get("id", str(i))
            entry = build_entry(qid, rec["question"], rec["answer"])
            fout.write(json.dumps(entry) + "\n")

    print(f"Saved → {output_path}")

###############################################
# 7. Example usage
###############################################

if __name__ == "__main__":
    import argparse
    from datasets import load_dataset

    parser = argparse.ArgumentParser(description="Build multi-granularity dataset")
    parser.add_argument("--start", type=int, default=0, help="Start index")
    parser.add_argument("--end", type=int, default=10, help="End index")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    args = parser.parse_args()

    # Auto-generate output filename if not provided
    if args.output is None:
        args.output = f"multi_granularity_gsm8k_{args.start}_{args.end}.jsonl"

    # example: load GSM8K
    ds = load_dataset("gsm8k", "main")["train"]  # change split if needed

    build_dataset_from_hf(
        input_dataset=ds.select(range(args.start, args.end)),
        output_path=args.output,
    )