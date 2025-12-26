import os, json, torch, re
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# ================================================================
# Load model once globally
# ================================================================
MODEL_NAME = "/home/lingjie7/models/huggingface/Qwen/Qwen3-0.6B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float16)
model.to(DEVICE)
model.eval()

# ================================================================
# Token counting helper (NEW)
# ================================================================
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

# ================================================================
# Simple Answer-first Dataset Builder
# ================================================================
def extract_final_answer(answer_text: str) -> str:
    """Extract the value after '####' in GSM8K-style answers."""
    match = re.search(r"####\s*(.+)", answer_text)
    if match:
        return match.group(1).strip()
    breakpoint()
    lines = [l.strip() for l in answer_text.split("\n") if l.strip()]
    return lines[-1] if lines else ""

def build_answer_first_dataset(dataset, output_path: str, max_new_tokens: int = 256):
    """
    Generate an answer-first reasoning dataset.
    For each record (id, question, answer):
        Extract answer after '####', then generate:
        "Ans: <answer>\n\n<reasoning>"
    """
    total_in_tokens = 0
    total_out_tokens = 0

    with open(output_path, "w") as fout:
        for i, rec in enumerate(tqdm(dataset, desc="Generating answer-first dataset")):
            qid = rec.get("id", str(i))
            question = rec["question"]
            full_answer_text = rec["answer"]
            extracted_answer = extract_final_answer(full_answer_text)

            prompt = f"""
You are solving a question and must first give the final answer, then a short reasoning.

QUESTION:
{question}

CORRECT FINAL ANSWER:
{extracted_answer}

FORMAT:
Begin with the exact line: "Ans: {extracted_answer}"
Then continue with a concise explanation of how to reach it.
""".strip()

            messages = [
                {"role": "system", "content": "You are a concise reasoning assistant."},
                {"role": "user", "content": prompt},
            ]
            chat_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )

            # ---- Run inference ----
            inputs = tokenizer(chat_text, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.2,
                    do_sample=False,
                    eos_token_id=tokenizer.eos_token_id,
                )

            # ---- Token counting (NEW) ----
            in_tok, out_tok = count_input_output_tokens(inputs, output_ids)
            total_in_tokens += in_tok
            total_out_tokens += out_tok

            generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
            text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            print(text)

            # ---- Save result ----
            entry = {
                "id": qid,
                "question": question,
                "answer_key": extracted_answer,
                "generated": text,
                "target_length": max_new_tokens,
                # (可选) 把token统计写进jsonl，方便后续分析
                "input_tokens": in_tok,
                "output_tokens": out_tok,
            }
            fout.write(json.dumps(entry) + "\n")

    print(f"Saved → {output_path}")
    print(f"Total input tokens : {total_in_tokens}")
    print(f"Total output tokens: {total_out_tokens}")
    if len(dataset) > 0:
        print(f"Avg input tokens   : {total_in_tokens / len(dataset):.2f}")
        print(f"Avg output tokens  : {total_out_tokens / len(dataset):.2f}")

# ================================================================
# Example use
# ================================================================
if __name__ == "__main__":
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main")["train"].select(range(10))

    build_answer_first_dataset(
        dataset=ds,
        output_path="answer_first_dataset.jsonl"
    )
