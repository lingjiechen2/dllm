"""
GRPO training for LLaDA with diffusion denoising.

Supported datasets: gsm8k, countdown, sudoku, math, code

Local users
-----------
- 1 GPU, quick test (no LoRA):
    accelerate launch \\
        --config_file scripts/accelerate_configs/ddp.yaml --num_processes 1 \\
        examples/llada/grpo.py \\
        --load_in_4bit True \\
        --dataset gsm8k --max_steps 50 \\
        --output_dir /path/to/output

- 8 GPUs, DeepSpeed ZeRO-2 (aligned with d1 paper settings):
    accelerate launch \\
        --config_file scripts/accelerate_configs/zero2.yaml \\
        examples/llada/grpo.py \\
        --load_in_4bit True --lora_r 128 --lora_alpha 64 --lora_dropout 0.05 \\
        --dataset gsm8k \\
        --num_train_epochs 10 --learning_rate 3e-6 \\
        --num_generations 6 --per_device_train_batch_size 6 \\
        --gradient_accumulation_steps 2 --num_iterations 12 \\
        --max_completion_length 256 --max_prompt_length 200 \\
        --block_size 32 --steps 128 \\
        --p_mask_prompt 0.15 --beta 0.04 --epsilon 0.5 \\
        --sync_ref_model True --ref_model_sync_steps 64 \\
        --output_dir /path/to/output

Slurm users
-----------
- 1 Node, 8 GPUs:
    sbatch --gres=gpu:8 scripts/train.slurm.sh \\
        --accelerate_config "zero2" \\
        --script_path "examples/llada/grpo.py" \\
        -- --dataset gsm8k --output_dir /path/to/output

Comparison with reference trainer (for functional equivalence testing):
    # Reference (d1/diffu-grpo):
    cd /path/to/d1/diffu-grpo && accelerate launch \\
        --config_file accelerate.yaml diffu_grpo_train.py \\
        --model_name_or_path GSAI-ML/LLaDA-8B-Instruct \\
        --dataset gsm8k --seed 42 --max_steps 50 \\
        --output_dir /tmp/ref_grpo

    # This script:
    cd /path/to/dllm && accelerate launch \\
        --config_file scripts/accelerate_configs/ddp.yaml --num_processes 1 \\
        examples/llada/grpo.py \\
        --load_in_4bit True \\
        --dataset gsm8k --seed 42 --max_steps 50 \\
        --output_dir /tmp/dllm_grpo
"""

import os
import random
from dataclasses import dataclass, field
from functools import partial
from typing import Optional

import numpy as np
import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from trl import ModelConfig, TrlParser

import dllm
from dllm.core.samplers import MDLMSampler, MDLMSamplerConfig
from dllm.pipelines.llada import DiffuGRPOConfig, DiffuGRPOTrainer
from dllm.utils.reward_funcs import (
    boxed_and_answer_tags_format_reward,
    coding_reward_func,
    correctness_reward_func,
    correctness_reward_func_math,
    countdown_reward_func,
    int_reward_func,
    soft_format_reward_func,
    strict_format_reward_func,
    sudoku_reward_func,
    xmlcount_reward_func,
)

logger = dllm.utils.get_default_logger(__name__)

# ---------------------------------------------------------------------------
# System prompts (matching the reference implementation)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

SUDOKU_SYSTEM_PROMPT = """
Please solve the following 4x4 Sudoku puzzle. The puzzle is provided as a 16-character string reading left-to-right, top-to-bottom, where '0' represents empty cells.

Rules:
- Fill empty cells with digits 1-4
- Each row must contain digits 1-4 exactly once
- Each column must contain digits 1-4 exactly once
- Each 2x2 box must contain digits 1-4 exactly once

Important: Your solution must be a COMPLETE 16-character string with only the digits 1-4, representing your final solved grid.

Respond in this exact format:
<reasoning>
Your step-by-step solving process
</reasoning>
<answer>
[16-character solution string with no spaces or separators]
</answer>
"""

# ---------------------------------------------------------------------------
# Dataset loaders (matching the reference implementation)
# ---------------------------------------------------------------------------


def _extract_hash_answer(text: str):
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def get_gsm8k_questions(split="train") -> Dataset:
    data = load_dataset("openai/gsm8k", "main")[split]
    return data.map(
        lambda x: {
            "prompt": [
                {"role": "user", "content": SYSTEM_PROMPT + "\n\n" + x["question"]}
            ],
            "answer": _extract_hash_answer(x["answer"]),
        }
    )


def get_countdown_questions(split="train") -> Dataset:
    data = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split=split)
    data = data.filter(lambda x: len(x["nums"]) == 3)
    return data.map(
        lambda x: {
            "prompt": [
                {
                    "role": "user",
                    "content": (
                        f"{SYSTEM_PROMPT}\nUsing only the numbers {x['nums']}, create an arithmetic "
                        f"expression that evaluates to exactly {x['target']}. You must use all numbers "
                        "from the list, and each number must be used exactly once. You may use the "
                        "operations +, -, *, and / as needed. After reasoning, provide only your final "
                        "expression inside <answer></answer> tags without including an equals sign or "
                        "the target number. For example, if the numbers are [2, 3, 4] and the target "
                        "is 5, a valid answer is: <answer>\n2*4-3\n</answer>"
                    ),
                }
            ],
            "target": x["target"],
            "numbers": x["nums"],
        }
    )


def get_sudoku_questions() -> Dataset:
    import pandas as pd

    sudoku_file_path = os.environ.get(
        "SUDOKU_DATA_PATH",
        "d1/dataset/4x4_sudoku_unique_puzzles.csv",
    )
    df = pd.read_csv(sudoku_file_path, dtype={"Puzzle": str, "Solution": str})
    data = Dataset.from_pandas(df)
    return data.map(
        lambda x: {
            "prompt": [
                {
                    "role": "user",
                    "content": f"{SUDOKU_SYSTEM_PROMPT}\n\nSolve the following Sudoku puzzle: {x['Puzzle']}\n",
                }
            ],
            "puzzle": x["Puzzle"],
            "solution": x["Solution"],
        }
    )


def get_math_questions(split="train") -> Dataset:
    data = load_dataset("ankner/math-500", split=split)
    return data.map(
        lambda x: {
            "prompt": [
                {
                    "role": "user",
                    "content": (
                        f"{SYSTEM_PROMPT}\n\nYou are a math expert. You will be given a question to "
                        f"solve. Solve it step by step. Wrap the final answer in a \\boxed{{}}. \n\n{x['problem']}"
                    ),
                }
            ],
            "answer": x["solution"],
        }
    )


def get_code_questions(split="train") -> Dataset:
    data = load_dataset("KodCode/KodCode-Light-RL-10K", split=split)
    data = data.train_test_split(test_size=0.1, seed=42)["train"]
    return data.map(
        lambda x: {
            "prompt": [
                {
                    "role": "user",
                    "content": (
                        f"{SYSTEM_PROMPT}\n\nYou are a coding expert. You will be given a coding problem "
                        f"to solve. Solve it step by step. \n\n{x['question']}"
                    ),
                }
            ],
            "answer": {"solution": x["solution"], "tests": x["test"]},
        }
    )


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


@dataclass
class TrainingArguments(DiffuGRPOConfig):
    output_dir: str = ".models/LLaDA-8B-Instruct/grpo"
    dataset: Optional[str] = field(
        default="gsm8k",
        metadata={"help": "Dataset to train on: gsm8k, countdown, sudoku, math, code."},
    )
    verbose_reward: bool = field(
        default=False,
        metadata={"help": "Enable verbose printing in reward functions."},
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def set_random_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = TrlParser((TrainingArguments, ModelConfig))
    training_args, model_config = parser.parse_args_and_config()

    if not model_config.model_name_or_path:
        model_config.model_name_or_path = "GSAI-ML/LLaDA-8B-Instruct"

    set_random_seed(training_args.seed)

    # ---- W&B project & run name -------------------------------------------------
    os.environ.setdefault("WANDB_PROJECT", "dlm-rl")
    if not training_args.run_name:
        model_tag = os.path.basename(model_config.model_name_or_path.rstrip("/"))
        training_args.run_name = f"{model_tag}-grpo-{training_args.dataset}"

    # ---- Dataset ----------------------------------------------------------------
    dataset_name = training_args.dataset
    if dataset_name == "gsm8k":
        dataset = get_gsm8k_questions("train")
        reward_functions = [
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            int_reward_func,
            correctness_reward_func,
        ]
    elif dataset_name == "countdown":
        dataset = get_countdown_questions("train")
        reward_functions = [countdown_reward_func]
    elif dataset_name == "sudoku":
        dataset = get_sudoku_questions()
        reward_functions = [sudoku_reward_func]
    elif dataset_name == "math":
        dataset = get_math_questions("train")
        reward_functions = [
            correctness_reward_func_math,
            boxed_and_answer_tags_format_reward,
        ]
    elif dataset_name == "code":
        dataset = get_code_questions("train")
        reward_functions = [xmlcount_reward_func, coding_reward_func]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if training_args.verbose_reward:
        reward_functions = [partial(fn, verbose=True) for fn in reward_functions]

    train_set = dataset.shuffle(seed=training_args.seed)

    # ---- Model & Tokenizer ------------------------------------------------------
    model_args = dllm.utils.ModelArguments(
        model_name_or_path=model_config.model_name_or_path,
        load_in_4bit=(
            model_config.load_in_4bit
            if hasattr(model_config, "load_in_4bit")
            else False
        ),
    )
    model = dllm.utils.get_model(model_args=model_args)
    tokenizer = dllm.utils.get_tokenizer(model_args=model_args)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.use_cache = False

    # ---- LoRA -------------------------------------------------------------------
    peft_config = None
    if model_config.lora_r and model_config.lora_r > 0:
        peft_config = LoraConfig(
            r=model_config.lora_r,
            lora_alpha=model_config.lora_alpha,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "up_proj",
                "down_proj",
                "gate_proj",
            ],
            task_type="CAUSAL_LM",
            lora_dropout=model_config.lora_dropout,
        )

    # ---- Sampler ----------------------------------------------------------------
    sampler = MDLMSampler(model=model, tokenizer=tokenizer)
    sampler_config = MDLMSamplerConfig(
        steps=training_args.steps,
        max_new_tokens=training_args.max_completion_length,
        block_size=training_args.block_size,
        temperature=training_args.temperature or 0.0,
        cfg_scale=training_args.cfg_scale,
        remasking=training_args.remasking,
    )

    # ---- Trainer ----------------------------------------------------------------
    logger.info("Start GRPO training...")
    trainer = DiffuGRPOTrainer(
        model=model,
        reward_funcs=reward_functions,
        args=training_args,
        train_dataset=train_set,
        processing_class=tokenizer,
        peft_config=peft_config,
        sampler=sampler,
        sampler_config=sampler_config,
    )

    if training_args.save_steps % training_args.num_iterations != 0:
        import warnings

        warnings.warn(
            f"save_steps ({training_args.save_steps}) is not divisible by "
            f"num_iterations ({training_args.num_iterations}). If resuming from a checkpoint, "
            f"you may need to manually pick a checkpoint where the step is divisible by "
            f"{training_args.num_iterations}."
        )

    trainer.train()


if __name__ == "__main__":
    main()
