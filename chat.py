"""
Interactive chat / sampling script for autoregressive models with visualization.

Examples
--------
# Chat with visualization
python -u chat.py --model_name_or_path "YOUR_MODEL_PATH"
"""

import sys
from dataclasses import dataclass
from typing import Optional

import torch
import transformers

import dllm
from sample import AutoregressiveSampler, AutoregressiveSamplerConfig


@dataclass
class ScriptArguments:
    model_name_or_path: str = "Qwen/Qwen3-0.6B"
    seed: int = 42
    chat_template: bool = True
    visualize: bool = True

    def __post_init__(self):
        self.model_name_or_path = dllm.utils.resolve_with_base_env(
            self.model_name_or_path, "BASE_MODELS_DIR"
        )


@dataclass
class ChatSamplerConfig(AutoregressiveSamplerConfig):
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: Optional[float] = 0.9
    stop_on_eos: bool = True


def main():
    parser = transformers.HfArgumentParser((ScriptArguments, ChatSamplerConfig))
    script_args, sampler_config = parser.parse_args_into_dataclasses()
    transformers.set_seed(script_args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    load_dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float32
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        dtype=load_dtype,
        trust_remote_code=True,
        device_map=None,
    ).to(device)
    model.eval()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        script_args.model_name_or_path,
        padding_side="right",
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token
    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    sampler = AutoregressiveSampler(model=model, tokenizer=tokenizer)

    if script_args.chat_template:
        dllm.utils.multi_turn_chat(
            sampler=sampler,
            sampler_config=sampler_config,
            visualize=script_args.visualize,
        )
    else:
        print("\nSingle-turn sampling (no chat template).")
        dllm.utils.single_turn_sampling(
            sampler=sampler,
            sampler_config=sampler_config,
            visualize=script_args.visualize,
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Bye!")
        sys.exit(0)
