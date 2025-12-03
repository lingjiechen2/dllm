"""
python -u examples/a2d/generate.py --model_name_or_path "YOUR_MODEL_PATH"
"""

from dataclasses import dataclass

import torch
import numpy as np
import transformers

import dllm


@dataclass
class ScriptArguments:
    model_name_or_path: str = "/mnt/lustrenew/mllm_aligned/shared/models/huggingface/Dream-org/Dream-v0-Instruct-7B"
    seed: int = 42
    visualize: bool = True

    def __post_init__(self):
        self.model_name_or_path = dllm.utils.resolve_with_base_env(
            self.model_name_or_path, "BASE_MODELS_DIR"
        )


@dataclass
class SamplerConfig(dllm.core.samplers.MDLMSamplerConfig):
    steps: int = 128
    max_new_tokens: int = 128
    block_size: int = 64
    temperature: float = 0.0
    remasking: str = "low_confidence"
    right_shift_logits: bool = True


parser = transformers.HfArgumentParser((ScriptArguments, SamplerConfig))
script_args, sampler_config = parser.parse_args_into_dataclasses()
transformers.set_seed(script_args.seed)

# Load model & tokenizer
model = dllm.utils.get_model(model_args=script_args).eval()
tokenizer = dllm.utils.get_tokenizer(model_args=script_args)
sampler = dllm.core.samplers.MDLMSampler(model=model, tokenizer=tokenizer)
terminal_visualizer = dllm.utils.TerminalVisualizer(tokenizer=tokenizer)

# --- Example 1: Batch sampling ---
print("\n" + "=" * 80)
print("TEST: sample()".center(80))
print("=" * 80)

messages = [
    # [{"role": "user", "content": "Introduce yourself to me."}],
    [{"role": "user", "content": "Write a solution to the following problem and make sure that it passes the tests:\n```python\nfrom typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n\n```\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nHere is the completed function:\n```python\nfrom typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n\n"}],
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
)
outputs = sampler.sample(inputs, sampler_config, return_dict=True)
sequences = dllm.utils.decode_trim(tokenizer, outputs.sequences.tolist(), inputs)

for iter, s in enumerate(sequences):
    print("\n" + "-" * 80)
    print(f"[Case {iter}]")
    print("-" * 80)
    print(s.strip() if s.strip() else "<empty>")
print("\n" + "=" * 80 + "\n")

# --- Analyze generation history ---
if hasattr(outputs, 'histories') and outputs.histories:
    print("\n" + "=" * 80)
    print("ANALYSIS: Generation History".center(80))
    print("=" * 80)
    
    history = outputs.histories
    MASK_ID = tokenizer.mask_token_id if hasattr(tokenizer, 'mask_token_id') else 151666
    
    steps = len(history)
    B, seqlen = history[0].shape
    
    # Compute first appearance step for each token
    device = history[0].device
    appear_step = torch.full((B, seqlen), -1, device=device)
    for t, step_tensor in enumerate(history):
        mask = (appear_step == -1) & (step_tensor != MASK_ID)
        appear_step[mask] = t
    nonzero_mask = appear_step != 0
    decoding_trace = appear_step[nonzero_mask]
    decoding_trace = decoding_trace.argsort()

    print(appear_step)
    print(decoding_trace)

breakpoint()

# if script_args.visualize:
#     terminal_visualizer.visualize(outputs.histories, rich=True)

from transformers import GenerationConfig

from huggingface_hub.utils._validators import validate_repo_id
