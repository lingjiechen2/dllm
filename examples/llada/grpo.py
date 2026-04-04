"""
GRPO training for LLaDA with diffusion denoising.

Local users
-----------
- 1 GPU (4bit + LoRA, useful for testing):
    accelerate launch \\
        --config_file scripts/accelerate_configs/ddp.yaml --num_processes 1 \\
        examples/llada/grpo.py \\
        --load_in_4bit True --lora True \\
        --dataset gsm8k --max_steps 10

- 8 GPUs (DeepSpeed ZeRO-2):
    accelerate launch \\
        --config_file scripts/accelerate_configs/zero2.yaml \\
        examples/llada/grpo.py \\
        --dataset gsm8k

Slurm users
-----------
- 1 Node, 8 GPUs:
    sbatch --gres=gpu:8 scripts/train.slurm.sh \\
        --accelerate_config "zero2" \\
        --script_path "examples/llada/grpo.py" \\
        -- --dataset gsm8k

Comparison with reference trainer (for functional equivalence testing):
    # Reference (d1/diffu-grpo):
    cd /home/lingjie7/d1/diffu-grpo && accelerate launch \\
        --config_file accelerate.yaml diffu_grpo_train.py \\
        --model_path GSAI-ML/LLaDA-8B-Instruct \\
        --dataset gsm8k --seed 42 --max_steps 10 \\
        --output_dir /tmp/ref_grpo

    # This script:
    cd /home/lingjie7/dllm && accelerate launch \\
        --config_file scripts/accelerate_configs/ddp.yaml --num_processes 1 \\
        examples/llada/grpo.py \\
        --dataset gsm8k --seed 42 --max_steps 10 \\
        --output_dir /tmp/dllm_grpo
"""

import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import transformers
from datasets import Dataset, load_dataset
from peft import LoraConfig
from trl import ModelConfig, TrlParser

import dllm
from dllm.core.samplers import MDLMSampler, MDLMSamplerConfig
from dllm.core.trainers import DiffuGRPOConfig, DiffuGRPOTrainer
from dllm.utils.reward_funcs import (
    coding_reward_func,
    correctness_reward_func,
    int_reward_func,
    soft_format_reward_func,
    strict_format_reward_func,
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
        metadata={"help": "Dataset to train on: gsm8k, code."},
    )
    model_path: Optional[str] = field(
        default="GSAI-ML/LLaDA-8B-Instruct",
        metadata={"help": "Path or HuggingFace ID of the model to train."},
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

    set_random_seed(training_args.seed)

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
    elif dataset_name == "code":
        dataset = get_code_questions("train")
        reward_functions = [xmlcount_reward_func, coding_reward_func]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    train_set = dataset.shuffle(seed=training_args.seed)

    # ---- Model & Tokenizer ------------------------------------------------------
    model_args = dllm.utils.ModelArguments(
        model_name_or_path=training_args.model_path,
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
