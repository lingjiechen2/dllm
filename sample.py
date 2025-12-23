"""
python -u sample.py --model_name_or_path "YOUR_AUTOREG_MODEL_PATH"
"""

from dataclasses import dataclass
from typing import Optional

import torch
import transformers

import dllm
from dllm.core.samplers.base import SamplerConfig, SamplerOutput


@dataclass
class ScriptArguments:
    model_name_or_path: str = "Qwen/Qwen3-0.6B"
    seed: int = 42
    visualize: bool = True

    def __post_init__(self):
        self.model_name_or_path = dllm.utils.resolve_with_base_env(
            self.model_name_or_path, "BASE_MODELS_DIR"
        )


@dataclass
class AutoregressiveSamplerConfig(SamplerConfig):
    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: Optional[float] = None
    stop_on_eos: bool = True


class AutoregressiveSampler:
    def __init__(self, model: transformers.PreTrainedModel, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.no_grad()
    def sample(
        self,
        inputs,
        config: AutoregressiveSamplerConfig | None = None,
        **kwargs,
    ) -> SamplerOutput | torch.Tensor:
        if config is None:
            config = AutoregressiveSamplerConfig()

        max_new_tokens = kwargs.get("max_new_tokens", config.max_new_tokens)
        temperature = kwargs.get("temperature", config.temperature)
        top_p = kwargs.get("top_p", config.top_p)
        stop_on_eos = kwargs.get("stop_on_eos", config.stop_on_eos)
        return_dict = kwargs.get("return_dict", config.return_dict)

        input_ids, attention_mask = self._normalize_inputs(inputs)
        device = input_ids.device

        eos_id = self.tokenizer.eos_token_id

        histories = [input_ids.clone()]

        cur_ids = input_ids
        cur_mask = attention_mask
        finished = torch.zeros(cur_ids.size(0), dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            logits = self.model(input_ids=cur_ids, attention_mask=cur_mask).logits
            next_token = self._select_token(
                logits[:, -1, :],
                temperature=temperature,
                top_p=top_p,
                finished=finished,
                eos_id=eos_id,
            )

            cur_ids = torch.cat([cur_ids, next_token], dim=1)
            cur_mask = torch.cat(
                [cur_mask, torch.ones_like(next_token, dtype=cur_mask.dtype)], dim=1
            )

            histories.append(cur_ids.clone())

            if eos_id is not None and stop_on_eos:
                finished = finished | (next_token.squeeze(-1) == eos_id)
                if torch.all(finished):
                    break

        sequences = cur_ids
        if not return_dict:
            return sequences
        return SamplerOutput(sequences=sequences, histories=histories)

    def _normalize_inputs(self, inputs):
        if isinstance(inputs, transformers.BatchEncoding) or isinstance(inputs, dict):
            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask")
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
        elif isinstance(inputs, torch.Tensor):
            input_ids = inputs
            attention_mask = torch.ones_like(
                input_ids, dtype=torch.long, device=input_ids.device
            )
        else:
            input_ids, attention_mask = self._pad_token_lists(inputs)

        if attention_mask.dtype != torch.long:
            attention_mask = attention_mask.long()

        device = self.model.device
        return input_ids.to(device), attention_mask.to(device)

    def _select_token(self, logits, temperature, top_p, finished, eos_id):
        if temperature is None or temperature <= 0.0:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            temp = max(temperature, 1e-5)
            probs = torch.softmax(logits / temp, dim=-1)
            if top_p is not None and 0.0 < top_p < 1.0:
                next_token = self._sample_top_p(probs, top_p)
            else:
                next_token = torch.multinomial(probs, num_samples=1)

        if eos_id is not None:
            eos_tensor = torch.full_like(next_token, eos_id)
            next_token = torch.where(finished.unsqueeze(1), eos_tensor, next_token)
        return next_token

    def _sample_top_p(self, probs, top_p: float):
        sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
        cum_probs = torch.cumsum(sorted_probs, dim=-1)
        cutoff = cum_probs > top_p
        cutoff[..., 1:] = cutoff[..., :-1].clone()
        cutoff[..., 0] = False
        sorted_probs = sorted_probs.masked_fill(cutoff, 0.0)
        denom = sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        sorted_probs = sorted_probs / denom
        next_idx = torch.multinomial(sorted_probs, num_samples=1)
        return sorted_idx.gather(-1, next_idx)

    def _pad_to_length(self, ids: torch.Tensor, target_len: int, pad_value: int):
        if ids.size(1) >= target_len:
            return ids[:, :target_len].detach().clone()
        pad_width = target_len - ids.size(1)
        pad = torch.full(
            (ids.size(0), pad_width),
            pad_value,
            device=ids.device,
            dtype=ids.dtype,
        )
        return torch.cat([ids, pad], dim=1).detach().clone()

    def _get_pad_value(self) -> int:
        for attr in ("mask_token_id", "pad_token_id", "eos_token_id"):
            val = getattr(self.tokenizer, attr, None)
            if val is not None:
                return val
        return 0

    def _pad_token_lists(self, inputs):
        device = self.model.device
        pad_val = self._get_pad_value()

        seq_tensors = []
        for seq in inputs:
            if isinstance(seq, torch.Tensor):
                t = seq.to(device)
            else:
                t = torch.as_tensor(seq, dtype=torch.long, device=device)
            seq_tensors.append(t)

        max_len = max(t.size(0) for t in seq_tensors) if seq_tensors else 0
        padded = []
        attn = []
        for t in seq_tensors:
            pad_len = max_len - t.size(0)
            if pad_len > 0:
                pad = torch.full(
                    (pad_len,), pad_val, device=device, dtype=t.dtype
                )
                t = torch.cat([t, pad], dim=0)
            padded.append(t)
            mask = torch.cat(
                [
                    torch.ones(t.size(0) - pad_len, device=device, dtype=torch.long),
                    torch.zeros(pad_len, device=device, dtype=torch.long),
                ]
            )
            attn.append(mask)

        if not padded:
            return torch.empty((0, 0), device=device, dtype=torch.long), torch.empty(
                (0, 0), device=device, dtype=torch.long
            )
        return torch.stack(padded, dim=0), torch.stack(attn, dim=0)


def main():
    parser = transformers.HfArgumentParser(
        (ScriptArguments, AutoregressiveSamplerConfig)
    )
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
    terminal_visualizer = dllm.utils.TerminalVisualizer(tokenizer=tokenizer)

    print("\n" + "=" * 80)
    print("TEST: autoregressive.sample()".center(80))
    print("=" * 80)

    messages = [
        # [{"role": "user", "content": "Write a Fibbonacci function in Python."}],
        [
            {
                "role": "user",
                "content": "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 10 kilometers per hour. How many kilometers can she run in 10 hours?",
            }
        ],
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        padding=True,
        enable_thinking=False,
    )

    outputs = sampler.sample(inputs, sampler_config, return_dict=True)

    input_ids_field = (
        inputs["input_ids"] if isinstance(inputs, (transformers.BatchEncoding, dict)) else inputs
    )
    input_ids_list = (
        input_ids_field.tolist() if isinstance(input_ids_field, torch.Tensor) else input_ids_field
    )

    sequences = dllm.utils.decode_trim(
        tokenizer,
        outputs.sequences.tolist(),
        input_ids_list,
    )

    # for idx, text in enumerate(sequences):
    #     print("\n" + "-" * 80)
    #     print(f"[Case {idx}]")
    #     print("-" * 80)
    #     print(text.strip() if text.strip() else "<empty>")
    # print("\n" + "=" * 80 + "\n")

    if script_args.visualize:
        terminal_visualizer.visualize(outputs.histories, rich=True)


if __name__ == "__main__":
    main()


# Lily can run 12 kilometers per hour for 4 hours. After that, she runs 10 kilometers per hour. How many kilometers can she run in 10 hours?
    
# Write a Fibbonacci function in Python.
