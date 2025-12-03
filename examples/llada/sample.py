"""
python -u examples/llada/sample.py --model_name_or_path "YOUR_MODEL_PATH"
"""

from dataclasses import dataclass

import transformers

import dllm


@dataclass
class ScriptArguments:
    model_name_or_path: str = "GSAI-ML/LLaDA-8B-Instruct"
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
    block_size: int = 32
    temperature: float = 0.0
    remasking: str = "low_confidence"


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
print("TEST: llada.sample()".center(80))
print("=" * 80)

messages = [
    [{"role": "user", "content": "Complete the following python code:\n\n\ndef truncate_number(number: float) -> float:\n    \"\"\" Given a positive floating point number, it can be decomposed into\n    and integer part (largest integer smaller than given number) and decimals\n    (leftover part always smaller than 1).\n\n    Return the decimal part of the number.\n    >>> truncate_number(3.5)\n    0.5\n    \"\"\""}],
    # [{"role": "user", "content": "Please write an educational python function."}],
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
breakpoint()
# if script_args.visualize:
#     terminal_visualizer.visualize(outputs.histories, rich=True)

# --- Example 2: Batch fill-in-the-blanks ---
print("\n" + "=" * 80)
print("TEST: llada.infilling()".center(80))
print("=" * 80)

masked_messages = [
    [
        {"role": "user", "content": tokenizer.mask_token * 20},
        {
            "role": "assistant",
            "content": "Sorry, I do not have answer to this question.",
        },
    ],
    [
        {"role": "user", "content": "Please write an educational python function."},
        {
            "role": "assistant",
            "content": "def hello_" + tokenizer.mask_token * 20 + " return",
        },
    ],
]

inputs = tokenizer.apply_chat_template(
    masked_messages,
    add_generation_prompt=False,
    tokenize=True,
)

outputs = sampler.infill(inputs, sampler_config, return_dict=True)
sequences = dllm.utils.decode_trim(tokenizer, outputs.sequences.tolist(), inputs)

for iter, (i, s) in enumerate(zip(inputs, sequences)):
    print("\n" + "-" * 80)
    print(f"[Case {iter}]")
    print("-" * 80)
    print("[Masked]:\n" + tokenizer.decode(i))
    print("\n[Filled]:\n" + (s.strip() if s.strip() else "<empty>"))
print("\n" + "=" * 80 + "\n")

if script_args.visualize:
    terminal_visualizer.visualize(outputs.histories, rich=True)
