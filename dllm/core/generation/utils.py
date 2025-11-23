import torch

from dllm.core.schedulers import BaseAlphaScheduler


def get_num_transfer_tokens(
    mask_index: torch.Tensor,
    steps: int,
    scheduler: BaseAlphaScheduler,
    stochastic: bool = False,
) -> torch.Tensor:
    mask_num = mask_index.sum(dim=1, keepdim=True)
    num_transfer_tokens = torch.zeros(
        mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64
    )
    for i in range(mask_num.size(0)):
        for t, s, j in zip(range(steps, 0, -1), range(steps - 1, -1, -1), range(steps)):
            s /= steps
            t /= steps
            reverse_transfer_prob = 1 - scheduler.reverse_mask_prob(s=s, t=t)
            if not stochastic:
                x = mask_num[i, 0].to(torch.float64) * reverse_transfer_prob
                num_transfer_tokens[i, j] = torch.round(x).to(torch.int64)
            else:
                n = mask_num[i, 0].to(torch.float64)
                num_transfer_tokens[i, j] = (
                    torch.distributions.Binomial(n, reverse_transfer_prob)
                    .sample()
                    .to(torch.int64)
                )
            num_transfer_tokens[i, j] = torch.minimum(
                num_transfer_tokens[i, j], mask_num[i, 0]
            )
            mask_num[i, 0] -= num_transfer_tokens[i, j]
            if mask_num[i, 0].item() == 0:
                break
    # Note: because llada is not conditioned on time, this allows us to skip steps with no unmasking (i.e. transfer).
    # Clear all zeros per row (compact) and right-pad with zeros
    # Remove zeros per row, then pad only up to the max length across rows
    rows = []
    max_len = 0
    for i in range(num_transfer_tokens.size(0)):
        nonzero = num_transfer_tokens[i][num_transfer_tokens[i] > 0]
        rows.append(nonzero)
        max_len = max(max_len, nonzero.numel())
    # Pad each row to max_len
    padded_rows = []
    for r in rows:
        if r.numel() < max_len:
            pad = torch.zeros(max_len - r.numel(), dtype=r.dtype, device=r.device)
            r = torch.cat([r, pad])
        padded_rows.append(r)
    return torch.stack(padded_rows, dim=0)


def decode_trim(tokenizer, seq_ids_list, input_ids_list) -> str:
    """
    Return only the generated text, truncated at the first EOS **after** the prompt.

    Args:
        tokenizer: HF tokenizer with eos_token_id / pad_token_id.
        seq_ids: Full sequence token ids from the model (prompt + generation).
        input_ids: The prompt token ids that were fed into the model.

    Behavior:
        - Finds the first eos_token_id that occurs at or after len(input_ids).
        - Slices generation up to (but not including) that EOS.
        - Decodes only the generation span, skipping special/pad tokens.
    """
    # Make sure we can index these
    sequences = []
    for seq_ids, input_ids in zip(seq_ids_list, input_ids_list):
        full = list(seq_ids)
        prompt = list(input_ids)

        # Skip left padding tokens (necessary for dream)
        pad_id = getattr(tokenizer, "pad_token_id", None)
        if pad_id is not None:
            while full and full[0] == pad_id:
                full.pop(0)

        start = len(prompt)
        end = len(full)

        eos_id = getattr(tokenizer, "eos_token_id", None)
        eot_id = getattr(tokenizer, "eot_token_id", None)
        if eos_id is not None:
            for i in range(start, len(full)):
                if full[i] in (eos_id, eot_id):
                    end = i
                    break

        gen_ids = full[start:end]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        # in case there is no eos_id or eot_id, just strings
        eos = getattr(tokenizer, "eos_token", None)
        eot = getattr(tokenizer, "eot_token", None)
        if eos:
            text = text.split(eos)[0]
        if eot:
            text = text.split(eot)[0]
        # return text.strip()
        sequences.append(text)
    return sequences
