"""
GRPO (Group Relative Policy Optimization) trainer for diffusion language models.

References:
  diffu-grpo: https://github.com/dllm-reasoning/d1/tree/main/diffu-grpo
  GRPO: https://arxiv.org/abs/2402.03300
"""

from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from accelerate.utils import set_seed
from datasets import Dataset, IterableDataset
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from trl import GRPOConfig
from trl.data_utils import is_conversational, maybe_apply_chat_template
from trl.extras.profiling import profiling_decorator
from trl.models import unwrap_model_for_generation
from trl.trainer.grpo_trainer import GRPOTrainer
from trl.trainer.utils import pad
from transformers import PreTrainedModel, PreTrainedTokenizerBase, TrainerCallback
from transformers.utils import is_peft_available
from trl.import_utils import is_vllm_available
from transformers.utils import is_rich_available

if is_peft_available():
    from peft import PeftConfig

RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


@dataclass
class DiffuGRPOConfig(GRPOConfig):
    """
    Configuration for DiffuGRPOTrainer, extending GRPOConfig with diffusion-specific parameters.
    """

    block_length: Optional[int] = field(
        default=64,
        metadata={"help": "Block length for block-wise diffusion generation."},
    )
    diffusion_steps: Optional[int] = field(
        default=64,
        metadata={"help": "Number of diffusion denoising steps per generation."},
    )
    cfg_scale: Optional[float] = field(
        default=0.0,
        metadata={"help": "Classifier-free guidance scale. 0.0 disables CFG."},
    )
    remasking: Optional[str] = field(
        default="low_confidence",
        metadata={"help": "Remasking strategy: 'low_confidence' or 'random'."},
    )
    p_mask_prompt: float = field(
        default=0.3,
        metadata={"help": "Probability of masking each prompt token in the forward process."},
    )
    mask_id: int = field(
        default=126336,
        metadata={"help": "Mask token id. Used as fallback if tokenizer has no mask_token_id."},
    )


class DiffuGRPOTrainer(GRPOTrainer):
    """
    GRPO Trainer adapted for masked diffusion language models.

    Overrides:
    - `_generate_and_score_completions`: replaces autoregressive generation with iterative denoising
    - `_get_per_token_logps`: replaces causal log-prob with diffusion forward-process log-prob

    All other GRPO infrastructure (advantage computation, PPO clipping, KL regularization,
    reference model management, buffering, distributed training) is inherited from TRL's GRPOTrainer.
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: Optional[DiffuGRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[
            Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]
        ] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[
            Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]
        ] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (
            None,
            None,
        ),
        peft_config: Optional["PeftConfig"] = None,
    ):
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
        )

    @property
    def _mask_id(self) -> int:
        """Resolve mask token id: prefer tokenizer's mask_token_id over config fallback."""
        if (
            self.processing_class is not None
            and hasattr(self.processing_class, "mask_token_id")
            and self.processing_class.mask_token_id is not None
        ):
            return self.processing_class.mask_token_id
        return self.args.mask_id

    # -----------------------------------------------------------------------
    # Diffusion generation utilities
    # -----------------------------------------------------------------------

    def add_gumbel_noise(self, logits, temperature, dtype):
        """
        Gumbel-max sampling for diffusion models.
        Per arXiv:2409.02908, low-precision Gumbel Max improves perplexity but reduces generation quality;
        we use float64 as in the reference implementation.
        """
        if temperature == 0.0:
            return logits
        logits = logits.to(dtype)
        noise = torch.rand_like(logits, dtype=dtype)
        gumbel_noise = (-torch.log(noise)) ** temperature
        return logits.exp() / gumbel_noise

    def generate(
        self,
        model,
        prompt,
        steps=128,
        gen_length=128,
        block_length=128,
        temperature=0.0,
        cfg_scale=0.0,
        remasking="low_confidence",
        mask_id=None,
    ):
        """Iterative denoising generation for masked diffusion language models."""
        if mask_id is None:
            mask_id = self._mask_id

        with torch.cuda.amp.autocast(enabled=True):
            bs = prompt.shape[0]
            dtype = model.dtype
            x = torch.full((bs, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
            x[:, : prompt.shape[1]] = prompt.clone()

            prompt_index = x != mask_id

            assert gen_length % block_length == 0
            num_blocks = gen_length // block_length
            steps_per_block = max(1, steps // num_blocks)

            for num_block in range(num_blocks):
                start_idx = prompt.shape[1] + num_block * block_length
                end_idx = prompt.shape[1] + (num_block + 1) * block_length

                block_mask_index = x[:, start_idx:end_idx] == mask_id
                num_transfer_tokens = self._get_num_transfer_tokens(block_mask_index, steps_per_block)

                for i in range(steps_per_block):
                    torch.cuda.empty_cache()
                    mask_index = x == mask_id

                    with torch.cuda.amp.autocast(enabled=self.args.fp16):
                        if cfg_scale > 0.0:
                            un_x = x.clone()
                            un_x[prompt_index] = mask_id
                            x_ = torch.cat([x, un_x], dim=0)
                            logits = model(x_).logits
                            logits, un_logits = torch.chunk(logits, 2, dim=0)
                            logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                        else:
                            logits = model(x).logits

                        logits_with_noise = self.add_gumbel_noise(logits, temperature=temperature, dtype=dtype)
                        x0 = torch.argmax(logits_with_noise, dim=-1)
                        del logits_with_noise

                        if remasking == "low_confidence":
                            p = F.softmax(logits.to(dtype), dim=-1)
                            x0_p = torch.squeeze(
                                torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                            )
                        elif remasking == "random":
                            x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                        else:
                            raise NotImplementedError(remasking)

                        x0_p[:, end_idx:] = -np.inf

                        x0 = torch.where(mask_index, x0, x)
                        confidence = torch.where(mask_index, x0_p, -np.inf)

                        transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                        for j in range(confidence.shape[0]):
                            num_tokens = num_transfer_tokens[j, i].item()
                            if num_tokens > 0:
                                _, select_index = torch.topk(confidence[j], k=num_tokens)
                                transfer_index[j, select_index] = True

                        x[transfer_index] = x0[transfer_index]
                        del x0, confidence, transfer_index

            return x

    def _forward_process(self, batch, prompt_index, mask_id, seed=None):
        """
        Apply the MDLM forward process (noising).

        - Prompt tokens are masked with probability p_mask_prompt.
        - Completion tokens are always masked.
        """
        if seed is not None:
            set_seed(seed)
        b, l = batch.shape
        t_p = torch.ones(b, device=batch.device) * self.args.p_mask_prompt

        random_matrix = torch.rand((b, l), device=batch.device)

        is_mask_prompt = prompt_index & (random_matrix < t_p.unsqueeze(1))
        is_mask_completion = ~prompt_index
        is_mask = is_mask_prompt | is_mask_completion

        noisy_batch = torch.where(is_mask, mask_id, batch)
        return noisy_batch

    def _get_model_logits(self, model, batch, prompt_index, cfg_scale, mask_id):
        """Forward pass with optional classifier-free guidance."""
        if cfg_scale > 0.0:
            assert len(prompt_index) == batch.shape[1]
            prompt_index_expanded = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            un_batch = batch.clone()
            un_batch[prompt_index_expanded] = mask_id
            batch = torch.cat([batch, un_batch])

        logits = model(batch).logits

        if cfg_scale > 0.0:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
        return logits

    def _get_num_transfer_tokens(self, mask_index, steps):
        """Precompute the number of tokens to unmask at each diffusion step."""
        mask_num = mask_index.sum(dim=1, keepdim=True)
        base = mask_num // steps
        remainder = mask_num % steps

        num_transfer_tokens = base.expand(-1, steps).clone()

        if remainder.sum() > 0:
            indices = torch.arange(steps, device=mask_index.device)
            mask = indices.unsqueeze(0) < remainder
            num_transfer_tokens[mask] += 1

        return num_transfer_tokens.to(torch.int64)

    # -----------------------------------------------------------------------
    # Override: per-token log probabilities (diffusion forward process)
    # -----------------------------------------------------------------------

    def _get_per_token_logps(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size=None,
    ) -> torch.Tensor:
        """
        Compute per-token log probabilities under the diffusion forward process.

        Replaces TRL's causal log-prob computation with:
        1. Apply random masking (prompt: p_mask_prompt, completion: always masked)
        2. Single bidirectional forward pass through the diffusion model
        3. Cross-entropy on completion tokens only → log-prob
        """
        batch_size = batch_size or input_ids.size(0)
        mask_id = self._mask_id
        seq_len = input_ids.size(1)
        prompt_length = seq_len - logits_to_keep

        prompt_index = torch.zeros(seq_len, dtype=torch.bool, device=input_ids.device)
        prompt_index[:prompt_length] = True

        all_logps = []
        for i in range(0, input_ids.size(0), batch_size):
            batch = input_ids[i : i + batch_size]

            # Apply diffusion forward process
            noisy_batch = self._forward_process(batch, prompt_index, mask_id)

            # Bidirectional forward pass (no attention_mask needed for full-sequence models)
            logits = self._get_model_logits(model, noisy_batch, prompt_index, self.args.cfg_scale, mask_id)

            # Compute log-prob for completion tokens only
            completion_logits = logits[:, -logits_to_keep:, :]
            completion_targets = batch[:, -logits_to_keep:]
            loss = F.cross_entropy(
                completion_logits.reshape(-1, completion_logits.size(-1)),
                completion_targets.reshape(-1),
                reduction="none",
            )
            logps = -loss.view(batch.size(0), logits_to_keep).to(torch.float32)
            all_logps.append(logps)

        return torch.cat(all_logps, dim=0)

    # -----------------------------------------------------------------------
    # Override: generation (diffusion instead of autoregressive)
    # -----------------------------------------------------------------------

    @profiling_decorator
    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        """
        Override TRL's _generate_and_score_completions to use diffusion generation.

        Only the generation block is replaced (iterative denoising instead of model.generate()).
        All downstream processing — reward computation, advantage normalization, logging —
        is handled by calling the parent's helper methods.
        """
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        prompt_inputs = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = super(GRPOTrainer, self)._prepare_inputs(prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]
            prompts_text = self.processing_class.batch_decode(
                prompt_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

        # ---- Diffusion generation (replaces model.generate) ----
        gen_length = self.args.max_completion_length
        block_length = self.args.block_length
        steps = self.args.diffusion_steps
        temperature = self.args.temperature or 0.0
        cfg_scale = self.args.cfg_scale
        mask_id = self._mask_id
        generation_batch_size = self.args.generation_batch_size or prompt_ids.size(0)

        with unwrap_model_for_generation(
            self.model_wrapped,
            self.accelerator,
            gather_deepspeed3_params=self.args.ds3_gather_for_generation,
        ) as unwrapped_model:
            with (
                FSDP.summon_full_params(self.model_wrapped, recurse=False)
                if self.is_fsdp_enabled
                else nullcontext()
            ):
                prompt_completion_ids_all = []
                for i in range(0, prompt_ids.size(0), generation_batch_size):
                    end_idx = min(i + generation_batch_size, prompt_ids.size(0))
                    batch_prompt_ids = prompt_ids[i:end_idx]
                    batch_result = self.generate(
                        model=unwrapped_model,
                        prompt=batch_prompt_ids,
                        steps=steps,
                        gen_length=gen_length,
                        block_length=block_length,
                        temperature=temperature,
                        cfg_scale=cfg_scale,
                        remasking=self.args.remasking,
                        mask_id=mask_id,
                    )
                    prompt_completion_ids_all.append(batch_result)
                    del batch_prompt_ids, batch_result
                    torch.cuda.empty_cache()

        prompt_completion_ids = torch.cat(prompt_completion_ids_all, dim=0)

        # Extract completion ids
        prompt_length = prompt_ids.size(1)
        prompt_ids = prompt_completion_ids[:, :prompt_length]
        completion_ids = prompt_completion_ids[:, prompt_length:]

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        completion_ids_list = [
            [id.item() for id, m in zip(row, mask_row) if m]
            for row, mask_row in zip(completion_ids, completion_mask)
        ]
        completion_lengths = completion_mask.sum(1)

        if self.mask_truncated_completions:
            truncated_completions = ~is_eos.any(dim=1)
            completion_mask = completion_mask * (~truncated_completions).unsqueeze(1).int()

        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size

        with torch.no_grad():
            if self.num_iterations > 1 or self.args.steps_per_generation > self.args.gradient_accumulation_steps:
                old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids, attention_mask, logits_to_keep, batch_size
                )
            else:
                old_per_token_logps = None

            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps = self._get_per_token_logps(
                        self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps = self._get_per_token_logps(
                            self.model, prompt_completion_ids, attention_mask, logits_to_keep
                        )
            else:
                ref_per_token_logps = None

        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        # Use TRL's reward computation helper
        rewards_per_func = self._calculate_rewards(inputs, prompts, completions, completion_ids_list)

        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        is_std_zero = torch.isclose(std_grouped_rewards, torch.zeros_like(std_grouped_rewards))

        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards
        if self.scale_rewards:
            advantages = advantages / (std_grouped_rewards + 1e-4)

        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        all_process_advantages = advantages.clone()
        advantages = advantages[process_slice]

        # Logging (matching TRL's metrics)
        if mode == "train":
            self.state.num_input_tokens_seen += self.accelerator.gather(attention_mask.sum()).sum().item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        from accelerate.utils import gather_object
        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())

        agg_terminated_with_eos = self.accelerator.gather(is_eos.any(dim=1))
        term_completion_lengths = agg_completion_lengths[agg_terminated_with_eos]
        clipped_completions_ratio = 1 - len(term_completion_lengths) / len(agg_completion_lengths)
        self._metrics[mode]["completions/clipped_ratio"].append(clipped_completions_ratio)
        if len(term_completion_lengths) == 0:
            term_completion_lengths = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completion_lengths.float().max().item())

        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            from trl.trainer.grpo_trainer import nanstd
            std_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_rewards)
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())
        self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())

        self._textual_logs["prompt"].extend(gather_object(prompts_text))
        self._textual_logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._textual_logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        self._textual_logs["advantages"].extend(all_process_advantages.tolist())

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
        }
