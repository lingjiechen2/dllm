import torch
from accelerate.utils import gather_object
from trl.trainer.grpo_trainer import nanstd


class GRPOMetrics:
    """
    Encapsulates all per-step metric writes for DiffuGRPOTrainer.

    Usage (mirrors MDLM's meter.update() pattern):
        self.meter.update(mode, ...)
    """

    def __init__(self, trainer):
        self.trainer = trainer

    def update(
        self,
        mode: str,
        attention_mask: torch.Tensor,
        completion_lengths: torch.Tensor,
        is_eos: torch.Tensor,
        rewards_per_func: torch.Tensor,
        mean_grouped_rewards: torch.Tensor,
        std_grouped_rewards: torch.Tensor,
        is_std_zero: torch.Tensor,
        prompts_text: list,
        completions_text: list,
        all_process_advantages: torch.Tensor,
    ) -> None:
        t = self.trainer
        acc = t.accelerator

        if mode == "train":
            t.state.num_input_tokens_seen += acc.gather(attention_mask.sum()).sum().item()
        t._metrics[mode]["num_tokens"] = [t.state.num_input_tokens_seen]

        agg_completion_lengths = acc.gather(completion_lengths)
        t._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
        t._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
        t._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())

        agg_terminated_with_eos = acc.gather(is_eos.any(dim=1))
        term_completion_lengths = agg_completion_lengths[agg_terminated_with_eos]
        t._metrics[mode]["completions/clipped_ratio"].append(1 - len(term_completion_lengths) / len(agg_completion_lengths))
        if len(term_completion_lengths) == 0:
            term_completion_lengths = torch.zeros(1, device=acc.device)
        t._metrics[mode]["completions/mean_terminated_length"].append(term_completion_lengths.float().mean().item())
        t._metrics[mode]["completions/min_terminated_length"].append(term_completion_lengths.float().min().item())
        t._metrics[mode]["completions/max_terminated_length"].append(term_completion_lengths.float().max().item())

        for i, name in enumerate(t.reward_func_names):
            t._metrics[mode][f"rewards/{name}/mean"].append(torch.nanmean(rewards_per_func[:, i]).item())
            t._metrics[mode][f"rewards/{name}/std"].append(nanstd(rewards_per_func[:, i]).item())
        t._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        t._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())
        t._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())

        t._textual_logs["prompt"].extend(gather_object(prompts_text))
        t._textual_logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(t.reward_func_names):
            t._textual_logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        t._textual_logs["advantages"].extend(all_process_advantages.tolist())
