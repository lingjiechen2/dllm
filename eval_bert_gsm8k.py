"""
python -u evaluate_gsm8k.py --model_name_or_path "YOUR_BERT_MODEL"
"""

from dataclasses import dataclass
import re
import json
from tqdm import tqdm
import transformers
import datasets

import dllm
from dllm.pipelines import llada
from dllm.tools.chat import decode_trim


# ----------------------------
# Arguments
# ----------------------------
@dataclass
class ScriptArguments:
    model_name_or_path: str = "/mnt/lustrenew/mllm_aligned/shared/models/tmp/ModernBERT-base/tulu-3-smoltalk/epochs-10-bs-384-len-1024/checkpoint-final"
    seed: int = 42
    max_samples: int = 1000


@dataclass
class GeneratorConfig(llada.LLaDAGeneratorConfig):
    steps: int = 128
    max_new_tokens: int = 128
    block_length: int = 64
    temperature: float = 0.0
    remasking: str = "low_confidence"


parser = transformers.HfArgumentParser((ScriptArguments, GeneratorConfig))
script_args, gen_config = parser.parse_args_into_dataclasses()
transformers.set_seed(script_args.seed)


# ----------------------------
# Load model + tokenizer
# ----------------------------
model = dllm.utils.get_model(model_args=script_args).eval()
tokenizer = dllm.utils.get_tokenizer(model_args=script_args)
generator = llada.LLaDAGenerator(model=model, tokenizer=tokenizer)


# ----------------------------
# Helper: extract numeric answer
# ----------------------------
def extract_number(text: str):
    m = re.findall(r"-?\d+", text)
    return int(m[-1]) if m else None


# ----------------------------
# Load GSM8K
# ----------------------------
gsm8k = datasets.load_dataset("gsm8k", "main")["test"]
gsm8k = gsm8k.select(range(min(script_args.max_samples, len(gsm8k))))


# ----------------------------
# Evaluation loop
# ----------------------------
correct = 0
log_path = "results_gsm8k.jsonl"
log_file = open(log_path, "w")

pbar = tqdm(range(len(gsm8k)), total=len(gsm8k), desc="Evaluating")

for idx in pbar:
    example = gsm8k[idx]

    question = example["question"]
    gt_answer = extract_number(example["answer"])
    messages = [[{"role": "user", "content": question}]]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
    )

    outputs = generator.generate(inputs, gen_config, return_dict_in_generate=True)
    pred_text = decode_trim(tokenizer, outputs.sequences.tolist(), inputs)[0].strip()
    pred_answer = extract_number(pred_text)

    # ---- Logging ----
    record = {
        "idx": idx,
        "question": question,
        "gt_answer": gt_answer,
        "pred_answer": pred_answer,
        "pred_text": pred_text,
    }
    log_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ---- Accuracy update ----
    if pred_answer == gt_answer:
        correct += 1

    current_acc = correct / (idx + 1)
    pbar.set_postfix({"acc": f"{current_acc:.4f}"})

log_file.close()
