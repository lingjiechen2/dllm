"""
GRPO (Group Relative Policy Optimization) trainer for diffusion language models.

References:
  diffu-grpo: https://github.com/dllm-reasoning/d1/tree/main/diffu-grpo
  GRPO: https://arxiv.org/abs/2402.03300
"""

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import torch
import torch.nn.functional as F
from accelerate.utils import set_seed
from datasets import Dataset, IterableDataset
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from trl import GRPOConfig
from trl.data_utils import is_conversational, maybe_apply_chat_template
from trl.extras.profiling import profiling_decorator
from trl.models import unwrap_model_for_generation
from trl.trainer.grpo_trainer import GRPOTrainer
from transformers import PreTrainedModel, PreTrainedTokenizerBase, TrainerCallback
from transformers.utils import is_peft_available

from accelerate.utils import gather_object
from trl.trainer.grpo_trainer import nanstd

from dllm.core.samplers import BaseSampler, BaseSamplerConfig, MDLMSampler, MDLMSamplerConfig

if is_peft_available():
    from peft import PeftConfig

RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


@dataclass
class DiffuGRPOConfig(GRPOConfig):
    """
    Configuration for DiffuGRPOTrainer, extending GRPOConfig with diffusion-specific parameters.
    """

    block_size: int = 64
    steps: int = 64
    cfg_scale: float = 0.0
    remasking: str = "low_confidence"
    p_mask_prompt: float = 0.3


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
        sampler: Optional[BaseSampler] = None,
        sampler_config: Optional[BaseSamplerConfig] = None,
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
        self.sampler = sampler or MDLMSampler(model=self.model, tokenizer=self.processing_class)
        self.sampler_config = sampler_config or MDLMSamplerConfig(
            steps=self.args.steps,
            max_new_tokens=self.args.max_completion_length,
            block_size=self.args.block_size,
            temperature=self.args.temperature or 0.0,
            cfg_scale=self.args.cfg_scale,
            remasking=self.args.remasking,
        )

    def _forward_process(self, batch, prompt_index, mask_id, seed=None):
        """
        Apply the MDLM forward process (noising).

        - Prompt tokens are masked with probability p_mask_prompt.
        - Completion tokens are always masked.
        """
        if seed is not None:
            set_seed(seed)
        is_mask_prompt = prompt_index & (torch.rand(batch.shape, device=batch.device) < self.args.p_mask_prompt)
        noised_input_ids = torch.where(is_mask_prompt | ~prompt_index, mask_id, batch)
        return noised_input_ids

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
        2. Single bidirectional forward pass through the diffusion model (with optional CFG)
        3. Cross-entropy on completion tokens only → log-prob
        """
        batch_size = batch_size or input_ids.size(0)
        mask_id = self.processing_class.mask_token_id
        cfg_scale = self.sampler_config.cfg_scale
        seq_len = input_ids.size(1)
        prompt_length = seq_len - logits_to_keep

        prompt_index = torch.arange(seq_len, device=input_ids.device) < prompt_length

        all_logps = []
        for i in range(0, input_ids.size(0), batch_size):
            batch = input_ids[i : i + batch_size]

            noised_input_ids = self._forward_process(batch, prompt_index, mask_id)

            if cfg_scale > 0.0:
                prompt_index_expanded = prompt_index.unsqueeze(0).repeat(noised_input_ids.shape[0], 1)
                un_batch = noised_input_ids.clone()
                un_batch[prompt_index_expanded] = mask_id
                logits, un_logits = torch.chunk(model(torch.cat([noised_input_ids, un_batch])).logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(noised_input_ids).logits

            completion_logits = logits[:, -logits_to_keep:, :]
            completion_targets = batch[:, -logits_to_keep:]
            loss = F.cross_entropy(
                completion_logits.reshape(-1, completion_logits.size(-1)),
                completion_targets.reshape(-1),
                reduction="none",
            )
            all_logps.append(-loss.view(batch.size(0), logits_to_keep).to(torch.float32))

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

        # ---- Diffusion generation via sampler ----
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
                self.sampler.model = unwrapped_model
                prompt_completion_ids_all = []
                for i in range(0, prompt_ids.size(0), generation_batch_size):
                    batch = list(prompt_ids[i:i + generation_batch_size])
                    out = self.sampler.sample(batch, self.sampler_config)
                    prompt_completion_ids_all.append(out)
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

        # ---- Metrics & logging ----
        acc = self.accelerator
        if mode == "train":
            self.state.num_input_tokens_seen += acc.gather(attention_mask.sum()).sum().item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        agg_completion_lengths = acc.gather(completion_lengths)
        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())

        agg_terminated_with_eos = acc.gather(is_eos.any(dim=1))
        term_completion_lengths = agg_completion_lengths[agg_terminated_with_eos]
        self._metrics[mode]["completions/clipped_ratio"].append(1 - len(term_completion_lengths) / len(agg_completion_lengths))
        if len(term_completion_lengths) == 0:
            term_completion_lengths = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completion_lengths.float().max().item())

        for i, name in enumerate(self.reward_func_names):
            self._metrics[mode][f"rewards/{name}/mean"].append(torch.nanmean(rewards_per_func[:, i]).item())
            self._metrics[mode][f"rewards/{name}/std"].append(nanstd(rewards_per_func[:, i]).item())
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
