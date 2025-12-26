import json
import os
import re
from typing import List, Dict
import datasets
from tqdm import tqdm

###############################################
# 1. Teacher model wrapper  (vLLM via OpenAI SDK)
###############################################

import os
from transformers import AutoTokenizer

# OpenAI-compatible client (works with vLLM server)
from openai import OpenAI

# Local path is still used for tokenizer + eval
MODEL_NAME = "/home/lingjie7/models/huggingface/Qwen/Qwen3-32B"

# vLLM/OpenAI-compatible settings (support both lowercase + uppercase env vars)
OPENAI_API_KEY = os.environ.get("openai_apikey") or os.environ.get("OPENAI_API_KEY") or "EMPTY"
VLLM_URL = os.environ.get("vllm_url") or os.environ.get("VLLM_URL") or "http://localhost:8000/v1"

# If you started vLLM with --served-model-name, put it here (otherwise defaults to MODEL_NAME)
SERVED_MODEL_NAME = os.environ.get("vllm_model") or os.environ.get("VLLM_MODEL") or MODEL_NAME

# Ensure base_url ends with /v1
if not VLLM_URL.rstrip("/").endswith("/v1"):
    VLLM_URL = VLLM_URL.rstrip("/") + "/v1"

client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=VLLM_URL,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def call_llm_with_tokens(prompt: str, max_new_tokens: int = 512, temperature: float = 0.2) -> tuple[str, int, int]:
    """
    Call vLLM (OpenAI-compatible) and return (decoded string, input_tokens, output_tokens).
    """
    messages = [
        {"role": "system", "content": "You are a helpful and concise assistant."},
        {"role": "user", "content": prompt},
    ]

    # Qwen3 thinking is enabled by default in vLLM; disable it via chat_template_kwargs
    # (matches your previous enable_thinking=False behavior)
    resp = client.chat.completions.create(
        model=SERVED_MODEL_NAME,
        messages=messages,
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=0.9,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )

    # Text
    text = (resp.choices[0].message.content or "").strip()

    # Token usage (vLLM returns OpenAI-style usage)
    in_tok = int(getattr(resp.usage, "prompt_tokens", 0) or 0)
    out_tok = int(getattr(resp.usage, "completion_tokens", 0) or 0)

    # Fallback if usage is missing for some reason
    if in_tok == 0:
        in_tok = len(tokenizer.encode(prompt, add_special_tokens=False))
    if out_tok == 0 and text:
        out_tok = len(tokenizer.encode(text, add_special_tokens=False))

    return text, in_tok, out_tok


# Keep old call_llm API unchanged
def call_llm(prompt: str, max_new_tokens: int = 512, temperature: float = 0.2) -> str:
    text, _, _ = call_llm_with_tokens(prompt, max_new_tokens=max_new_tokens, temperature=temperature)
    return text


# =========================
# NEW: token counting helper
# =========================
def count_input_output_tokens(inputs, output_ids) -> tuple[int, int]:
    """
    inputs: tokenizer(...) 的返回（batch=1）
    output_ids: model.generate(...) 的返回（batch=1）
    返回 (input_tokens, output_tokens)
    """
    input_len = int(inputs["input_ids"].shape[1])
    total_len = int(output_ids.shape[1])
    output_len = max(total_len - input_len, 0)
    return input_len, output_len

###############################################
# 2. Templates for each granularity
###############################################

GRANULARITY_SPECS = {
    32:  {"name": "MVA",        "instruction": "Provide ONLY `Ans: <value>` in ≤32 tokens."},
    64:  {"name": "WHY+ANS",    "instruction": "Write a 1-sentence brief reasoning, then the final line `Ans: <value>` (≤64 tokens)."},
    128: {"name": "BULLETS3",   "instruction": "Give exactly 3 bullets, each ≤20 tokens, ending with `Ans: <value>`."},
    256: {"name": "SHORT_COT",  "instruction": "Give a short reasoning ≤8 steps, then `Ans: <value>`."},
    512: {"name": "COT+VERIFY", "instruction": "Give detailed step-by-step reasoning (≥12 numbered steps), then a 1-line verification, then the final line `Ans: <value>`."},
    1024:{"name": "FULL_COT",   "instruction": "Give full, detailed reasoning (≥20 numbered steps) plus a thorough explanation (4–8 paragraphs), then end with the final line `Ans: <value>`."}
}

###############################################
# 3. Build compression prompt (no facts)
###############################################

def extract_final_answer_value(answer: str) -> str:
    """
    Extract the final answer value from common dataset formats (e.g. GSM8K uses '#### <value>').
    Falls back to the last non-empty line.
    """
    m = re.search(r"####\s*(.+)$", answer, flags=re.MULTILINE)
    if m:
        return m.group(1).strip()

    m = re.search(r"^\s*Ans:\s*(.+)$", answer, flags=re.IGNORECASE | re.MULTILINE)
    if m:
        return m.group(1).strip()

    lines = [ln.strip() for ln in answer.splitlines() if ln.strip()]
    return lines[-1] if lines else answer.strip()


def build_compression_prompt(question: str, answer: str, L: int) -> str:
    spec = GRANULARITY_SPECS[L]
    final_answer_value = extract_final_answer_value(answer)

    small_budget_hint = ""
    if L <= 32:
        small_budget_hint = (
            "- For this very small budget, provide **only** the final answer line.\n"
            "- Do not include any reasoning, explanation, or extra words.\n"
        )

    ans_why_hint = ""
    if L == 64:
        ans_why_hint = (
            "- Use exactly 2 lines and nothing else:\n"
            "- Line 1: 1 sentence of brief reasoning (no label like 'Why:'; do NOT start with 'Ans:').\n"
            f"- Line 2: exactly `Ans: {final_answer_value}`\n"
            "- Do NOT use angle brackets, XML tags, or placeholders like '<value>'.\n"
        )

    long_budget_hint = ""
    if L == 512:
        long_budget_hint = (
            "- Be detailed and use the budget (do not be terse):\n"
            "- Include ≥12 numbered reasoning steps with intermediate calculations.\n"
            "- Add a single verification line starting with `Verify:`.\n"
            f"- Aim for roughly 300–450 tokens (still ≤ {L}).\n"
        )
    elif L == 1024:
        long_budget_hint = (
            "- Be very detailed and use the budget (do not be terse):\n"
            "- Include ≥20 numbered reasoning steps, then 4–8 explanatory paragraphs.\n"
            "- Add a short verification section (2–3 lines).\n"
            f"- Aim for roughly 600–900 tokens (still ≤ {L}).\n"
        )

    return f"""
You are generating an answer under a strict token budget.

QUESTION:
{question}

FINAL ANSWER VALUE (must use exactly in the last line):
{final_answer_value}

REFERENCE ANSWER KEY (may include extra text; do NOT copy verbatim):
{answer}

TARGET FORMAT ({spec['name']}):
{spec['instruction']}

RULES:
- Must be ≤ {L} tokens.
{small_budget_hint}{ans_why_hint}{long_budget_hint}- If reasoning or explanation is allowed, it should come **before** the final answer.
- The final answer must appear exactly once, **at the end**, as exactly: `Ans: {final_answer_value}`.
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
    tokens = eval_tokenizer.encode(text, add_special_tokens=False)
    token_len = len(tokens)
    within_budget = token_len <= L

    contains_final_answer = answer.strip().lower() in text.lower()

    extracted = ""
    if "ans:" in text.lower():
        try:
            extracted = text.lower().split("ans:")[-1].strip().split()[0]
        except IndexError:
            extracted = ""
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

        # NEW: 用带 token 统计的版本（不改生成设置）
        text, in_tok, out_tok = call_llm_with_tokens(prompt, max_new_tokens=L)

        metrics = evaluate_generation(text, answer, L)
        results.append({
            "L": L,
            "format": GRANULARITY_SPECS[L]["name"],
            "text": text,
            # NEW: 把模型实际消耗的 tokens 记录下来
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            # （可选）如果你也想把指标写进去，可以取消注释
            # "metrics": metrics,
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
    total_in_tokens = 0
    total_out_tokens = 0
    total_calls = 0

    with open(output_path, "w") as fout:
        for i, rec in enumerate(tqdm(input_dataset, desc="Processing questions")):
            qid = rec.get("id", str(i))
            entry = build_entry(qid, rec["question"], rec["answer"])

            # NEW: 累计每个预算一次生成的 token 使用量
            for b in entry["budgets"]:
                total_in_tokens += int(b.get("input_tokens", 0))
                total_out_tokens += int(b.get("output_tokens", 0))
                total_calls += 1

            fout.write(json.dumps(entry) + "\n")

    print(f"Saved → {output_path}")
    print(f"Total input tokens : {total_in_tokens}")
    print(f"Total output tokens: {total_out_tokens}")
    if total_calls > 0:
        print(f"Avg input tokens/call : {total_in_tokens / total_calls:.2f}")
        print(f"Avg output tokens/call: {total_out_tokens / total_calls:.2f}")

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

    if args.output is None:
        args.output = f"multi_granularity_gsm8k_{args.start}_{args.end}.jsonl"

    ds = load_dataset("gsm8k", "main")["train"]

    build_dataset_from_hf(
        input_dataset=ds.select(range(args.start, args.end)),
        output_path=args.output,
    )
