# """
# LLaDA / MoE / Dream attention mask invariance tests (compact version)
# """

# import gc

# import torch
# import transformers
# import dllm
# import pytest

# ERROR_THRESHOLD = 1e-3


# def _cuda_cleanup():
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#         # Reclaim interprocess memory blocks (useful after large model del)
#         try:
#             torch.cuda.ipc_collect()
#         except Exception:
#             # Not all PyTorch builds expose ipc_collect on all platforms
#             pass


# def _forward_variants(model):
#     """
#     Run the 5 padding/mask variants and return tensors sliced to the 'real' tokens [101,102,103,104].
#     Returns dict: {'A','B','C','D','E'} each [1, 4, H]
#     """
#     device = model.device

#     # base token sequence (the "real" tokens)
#     base_token = torch.tensor([[101, 102, 103, 104]], device=device)
#     pad_token = torch.tensor([[0]], device=device)

#     # A: no padding
#     a_ids = base_token
#     a_mask = torch.ones_like(a_ids)

#     # B: left-pad a 0
#     b_ids = torch.cat([pad_token, base_token], dim=1)
#     b_mask = torch.cat(
#         [torch.zeros_like(pad_token), torch.ones_like(base_token)], dim=1
#     )

#     # C: right-pad a 0
#     c_ids = torch.cat([base_token, pad_token], dim=1)
#     c_mask = torch.cat(
#         [torch.ones_like(base_token), torch.zeros_like(pad_token)], dim=1
#     )

#     # D: same as A but attention_mask=None
#     d_ids = base_token
#     d_mask = None

#     # E: same as A but omit attention_mask entirely
#     e_ids = base_token

#     with torch.no_grad():
#         out_A = model(input_ids=a_ids, attention_mask=a_mask).logits  # [1,4,H]
#         out_B = model(input_ids=b_ids, attention_mask=b_mask).logits[:, 1:]  # [1,4,H]
#         out_C = model(input_ids=c_ids, attention_mask=c_mask).logits[:, :-1]  # [1,4,H]
#         out_D = model(input_ids=d_ids, attention_mask=d_mask).logits  # [1,4,H]
#         out_E = model(input_ids=e_ids).logits  # [1,4,H]

#     return {"A": out_A, "B": out_B, "C": out_C, "D": out_D, "E": out_E}


# def _forward_batched(model):
#     """
#     Run A/B/C in a single batch and slice back to the 'real' tokens [101,102,103,104].
#     Returns dict: {'A','B','C'} each [1, 4, H]
#     """
#     device = model.device

#     base_token = torch.tensor([[101, 102, 103, 104]], device=device)
#     pad_token = torch.tensor([[0]], device=device)

#     # To batch them, make all seq_len = 5
#     # A (no pad): right-pad with 0, mask last position as 0
#     a_ids = torch.cat([base_token, pad_token], dim=1)  # [1,5]
#     a_mask = torch.cat(
#         [torch.ones_like(base_token), torch.zeros_like(pad_token)], dim=1
#     )

#     # B (left pad): same as single variant
#     b_ids = torch.cat([pad_token, base_token], dim=1)  # [1,5]
#     b_mask = torch.cat(
#         [torch.zeros_like(pad_token), torch.ones_like(base_token)], dim=1
#     )

#     # C (right pad): same as single variant
#     c_ids = torch.cat([base_token, pad_token], dim=1)  # [1,5]
#     c_mask = torch.cat(
#         [torch.ones_like(base_token), torch.zeros_like(pad_token)], dim=1
#     )

#     batch_ids = torch.cat([a_ids, b_ids, c_ids], dim=0)   # [3,5]
#     batch_mask = torch.cat([a_mask, b_mask, c_mask], dim=0)  # [3,5]

#     with torch.no_grad():
#         logits = model(input_ids=batch_ids, attention_mask=batch_mask).logits  # [3,5,H]

#     # Recover the "real" tokens [1,2,3,4] view for each case
#     out_A = logits[0:1, :4, :]   # A: positions 0..3
#     out_B = logits[1:2, 1:, :]   # B: positions 1..4
#     out_C = logits[2:3, :4, :]   # C: positions 0..3

#     return {"A": out_A, "B": out_B, "C": out_C}


# def _assert_invariance(outs: dict, tag: str):
#     ref = outs["A"]
#     for k in ("B", "C", "D", "E"):
#         assert torch.allclose(
#             ref, outs[k], atol=ERROR_THRESHOLD, rtol=ERROR_THRESHOLD
#         ), f"[{tag}] Mismatch A vs {k}"


# def _assert_batch_consistency(single_outs: dict, batch_outs: dict, tag: str):
#     """
#     Check that running A/B/C separately vs batched gives the same logits
#     on the 'real' tokens.
#     """
#     for k in ("A", "B", "C"):
#         assert torch.allclose(
#             single_outs[k], batch_outs[k], atol=ERROR_THRESHOLD, rtol=ERROR_THRESHOLD
#         ), f"[{tag}] Batch vs single mismatch for {k}"


# @pytest.mark.parametrize(
#     "repo, attn_impl, human_name",
#     [
#         ("GSAI-ML/LLaDA-8B-Base", None, "LLaDA Base"),
#         # ("inclusionAI/LLaDA-MoE-7B-A1B-Base", None, "LLaDA MoE"),
#         # ("Dream-org/Dream-v0-Base-7B", None, "Dream Base"),
#     ],
# )
# def test_attention_mask_invariance(repo, attn_impl, human_name):
#     """
#     For each model/backend:
#       1) Check padding/mask invariance across A..E on the 'real' tokens.
#       2) Check that A/B/C single-sample vs batched inference produce identical logits.
#       3) Print a ✅ message for debug visibility (pytest still enforces assertions).
#     """
#     model_path = dllm.utils.resolve_with_base_env(repo, "BASE_MODELS_DIR")

#     if attn_impl is None:
#         model = transformers.AutoModel.from_pretrained(
#             model_path, dtype=torch.float32, device_map="auto"
#         ).eval()
#     else:
#         config = transformers.AutoConfig.from_pretrained(
#             model_path, attn_implementation=attn_impl
#         )
#         model = transformers.AutoModel.from_pretrained(
#             model_path, config=config, dtype=torch.float32, device_map="auto"
#         ).eval()

#     outs_single = _forward_variants(model)
#     _assert_invariance(outs_single, human_name)

#     outs_batch = _forward_batched(model)
#     _assert_batch_consistency(outs_single, outs_batch, human_name)

#     print(
#         f"✅ {human_name} attention mask invariance + batch consistency passed within {ERROR_THRESHOLD}."
#     )
#     del model
#     gc.collect()
#     _cuda_cleanup()

"""
LLaDA / MoE / Dream attention mask invariance tests

This script checks:

1) Single-sample padding/mask invariance for multiple base token sequences.
   For each base token sequence, we create 5 variants:

   - "no_padding":    [t1, t2, t3, t4],    mask [1,1,1,1]
   - "left_padding":  [0, t1, t2, t3, t4], mask [0,1,1,1,1]
   - "right_padding": [t1, t2, t3, t4, 0], mask [1,1,1,1,0]
   - "no_mask":       [t1, t2, t3, t4],    mask=None
   - "mask_omitted":  [t1, t2, t3, t4],    attention_mask not passed

   All must produce identical logits on the 4 "real" tokens.

2) Batch vs single consistency:
   Given multiple base sequences, we test:

   (a) No-padding batch:
       stack all base sequences in a batch:
         [[t1_0..3],          # base set 0
          [t1_0..3],          # base set 1
          ...]
       → each row's logits must match the corresponding single-sample "no_padding".

   (b) Padded batch:
       for each base sequence, create two rows:
         right-padded: [t1..t4, 0], mask [1,1,1,1,0]
         left-padded:  [0, t1..t4], mask [0,1,1,1,1]
       batch size = 2 * num_base_sets.
       On the real tokens:
         right-padded rows: positions 0..3
         left-padded rows:  positions 1..4
       → all must match the corresponding single-sample "no_padding".
"""

import gc
from typing import Dict, List

import torch
import transformers
import dllm
import pytest

# Numerical tolerance
ERROR_THRESHOLD = 1e-3

# A list of base token sequences to test.
# You can add more sequences if needed.
BASE_TOKEN_SETS: List[List[int]] = [
    [101, 102, 103, 104],
    [201, 202, 203, 204],
]

# Padding token ID (adjust if your models use a different pad ID)
PAD_TOKEN_ID = 0


def _cuda_cleanup():
    """Free CUDA memory between tests."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            # Not all PyTorch builds expose ipc_collect
            pass


def _get_model_device(model: torch.nn.Module) -> torch.device:
    """
    Get a device for creating input tensors.

    NOTE: with device_map="auto", parameters may be sharded across devices.
    In most HF setups, you can still create inputs on the first parameter's device.
    If your setup differs, adjust this helper accordingly.
    """
    return next(model.parameters()).device


# -------------------------------------------------------------------------
# 1) Single-sample variants over multiple base token sets
# -------------------------------------------------------------------------
def _forward_variants(model) -> Dict[str, torch.Tensor]:
    """
    For each base token set in BASE_TOKEN_SETS, run 5 variants:

      "no_padding"
      "left_padding"
      "right_padding"
      "no_mask"
      "mask_omitted"

    For each variant we only keep logits on the 4 real token positions.

    Returns:
        dict mapping variant_name -> logits tensor of shape [N, 4, H],
        where N = len(BASE_TOKEN_SETS), in the same order as BASE_TOKEN_SETS.
    """
    device = _get_model_device(model)

    # Accumulators for each variant. We will cat along dim=0 at the end.
    acc = {
        "no_padding": [],
        "left_padding": [],
        "right_padding": [],
        "no_mask": [],
        "mask_omitted": [],
    }

    for base_tokens in BASE_TOKEN_SETS:
        base = torch.tensor([base_tokens], device=device)         # [1,4]
        pad = torch.tensor([[PAD_TOKEN_ID]], device=device)       # [1,1]

        # no_padding
        ids_no_pad = base                                        # [1,4]
        mask_no_pad = torch.ones_like(ids_no_pad)                # [1,4]

        # left_padding: [0, t1, t2, t3, t4]
        ids_left = torch.cat([pad, base], dim=1)                 # [1,5]
        mask_left = torch.cat(
            [torch.zeros_like(pad), torch.ones_like(base)], dim=1
        )                                                        # [1,5]

        # right_padding: [t1, t2, t3, t4, 0]
        ids_right = torch.cat([base, pad], dim=1)                # [1,5]
        mask_right = torch.cat(
            [torch.ones_like(base), torch.zeros_like(pad)], dim=1
        )                                                        # [1,5]

        # no_mask: attention_mask=None
        ids_no_mask = base
        mask_none = None

        # mask_omitted: do not pass attention_mask at all
        ids_omitted = base

        with torch.no_grad():
            out_no_pad = model(
                input_ids=ids_no_pad, attention_mask=mask_no_pad
            ).logits  # [1,4,H]

            out_left = model(
                input_ids=ids_left, attention_mask=mask_left
            ).logits[:, 1:]  # [1,4,H] (skip pad position)

            out_right = model(
                input_ids=ids_right, attention_mask=mask_right
            ).logits[:, :-1]  # [1,4,H] (ignore padded position)

            out_no_mask = model(
                input_ids=ids_no_mask, attention_mask=mask_none
            ).logits  # [1,4,H]

            out_omitted = model(input_ids=ids_omitted).logits  # [1,4,H]

        acc["no_padding"].append(out_no_pad)
        acc["left_padding"].append(out_left)
        acc["right_padding"].append(out_right)
        acc["no_mask"].append(out_no_mask)
        acc["mask_omitted"].append(out_omitted)

    # Concatenate results for each variant along batch axis.
    outs = {
        key: torch.cat(tensors, dim=0)  # [N,4,H]
        for key, tensors in acc.items()
    }
    return outs


def _assert_invariance(outs: Dict[str, torch.Tensor], tag: str):
    """
    Check that for all base token sets, all variants match "no_padding".
    """
    ref = outs["no_padding"]  # [N,4,H]
    for key in ("left_padding", "right_padding", "no_mask", "mask_omitted"):
        assert torch.allclose(
            ref, outs[key], atol=ERROR_THRESHOLD, rtol=ERROR_THRESHOLD
        ), f"[{tag}] Single-sample mismatch: no_padding vs {key}"


# -------------------------------------------------------------------------
# 2) Batch tests over all base token sets
# -------------------------------------------------------------------------
def _forward_batch_nopad(model) -> torch.Tensor:
    """
    Batch = stack all base token sets (no padding):

        [
          BASE_TOKEN_SETS[0],  # [t1, t2, t3, t4]
          BASE_TOKEN_SETS[1],
          ...
        ]

    attention_mask = 1 for all positions.

    Returns:
        logits: [N, 4, H] in the same order as BASE_TOKEN_SETS.
    """
    device = _get_model_device(model)

    base_batch = torch.tensor(BASE_TOKEN_SETS, device=device)  # [N,4]
    mask = torch.ones_like(base_batch)                         # [N,4]

    with torch.no_grad():
        logits = model(input_ids=base_batch, attention_mask=mask).logits  # [N,4,H]

    return logits


def _forward_batch_padded(model) -> torch.Tensor:
    """
    Padded batch over all base token sets.

    For each base token set base[i] = [t1, t2, t3, t4], we create:

      right-padded row i_r: [t1,t2,t3,t4,0], mask [1,1,1,1,0]
      left-padded  row i_l: [0,t1,t2,t3,t4], mask [0,1,1,1,1]

    We then interleave them in the batch as:

      row 0: base[0] right-padded
      row 1: base[0] left-padded
      row 2: base[1] right-padded
      row 3: base[1] left-padded
      ...

    So the batch size is 2 * N.

    Returns:
        logits: [2N, 5, H]
    """
    device = _get_model_device(model)

    base_batch = torch.tensor(BASE_TOKEN_SETS, device=device)  # [N,4]
    N = base_batch.size(0)

    pad_col = torch.full((N, 1), PAD_TOKEN_ID, device=device)  # [N,1]

    # Right-padded: [t1..t4, 0]
    ids_right = torch.cat([base_batch, pad_col], dim=1)        # [N,5]
    mask_right = torch.cat(
        [torch.ones_like(base_batch), torch.zeros_like(pad_col)], dim=1
    )                                                          # [N,5]

    # Left-padded: [0, t1..t4]
    ids_left = torch.cat([pad_col, base_batch], dim=1)         # [N,5]
    mask_left = torch.cat(
        [torch.zeros_like(pad_col), torch.ones_like(base_batch)], dim=1
    )                                                          # [N,5]

    # Interleave right/left per base:
    # shape [N, 2, 5] -> reshape to [2N, 5]
    ids_stacked = torch.stack([ids_right, ids_left], dim=1)    # [N,2,5]
    mask_stacked = torch.stack([mask_right, mask_left], dim=1) # [N,2,5]

    batch_ids = ids_stacked.reshape(-1, ids_stacked.size(-1))   # [2N,5]
    batch_mask = mask_stacked.reshape(-1, mask_stacked.size(-1))# [2N,5]

    with torch.no_grad():
        logits = model(input_ids=batch_ids, attention_mask=batch_mask).logits  # [2N,5,H]

    return logits


def _assert_batch_equal_to_single(
    single_no_pad: torch.Tensor,
    batch_nopad: torch.Tensor,
    batch_padded: torch.Tensor,
    tag: str,
):
    """
    Compare batch outputs to single-sample "no_padding" for each base token set.

    Args:
        single_no_pad: [N,4,H] from _forward_variants()["no_padding"]
        batch_nopad:   [N,4,H] from _forward_batch_nopad
        batch_padded:  [2N,5,H] from _forward_batch_padded
    """
    N = single_no_pad.size(0)

    for i in range(N):
        ref = single_no_pad[i : i + 1]  # [1,4,H]
        tokens = BASE_TOKEN_SETS[i]

        # 1) No-padding batch row i
        assert torch.allclose(
            ref,
            batch_nopad[i : i + 1, :, :],
            atol=ERROR_THRESHOLD,
            rtol=ERROR_THRESHOLD,
        ), (
            f"[{tag}] no-pad batch mismatch for base index {i}, "
            f"tokens={tokens}"
        )

        # 2) Padded batch right-padded row (index 2*i): positions 0..3
        assert torch.allclose(
            ref,
            batch_padded[2 * i : 2 * i + 1, :4, :],
            atol=ERROR_THRESHOLD,
            rtol=ERROR_THRESHOLD,
        ), (
            f"[{tag}] padded batch RIGHT mismatch for base index {i}, "
            f"tokens={tokens} (positions 0..3)"
        )

        # 3) Padded batch left-padded row (index 2*i+1): positions 1..4
        assert torch.allclose(
            ref,
            batch_padded[2 * i + 1 : 2 * i + 2, 1:, :],
            atol=ERROR_THRESHOLD,
            rtol=ERROR_THRESHOLD,
        ), (
            f"[{tag}] padded batch LEFT mismatch for base index {i}, "
            f"tokens={tokens} (positions 1..4)"
        )


# -------------------------------------------------------------------------
# PyTest entry point
# -------------------------------------------------------------------------
@pytest.mark.parametrize(
    "repo, attn_impl, human_name",
    [
        ("GSAI-ML/LLaDA-8B-Base", None, "LLaDA Base"),
        ("inclusionAI/LLaDA-MoE-7B-A1B-Base", None, "LLaDA MoE"),
        ("Dream-org/Dream-v0-Base-7B", None, "Dream Base"),
    ],
)
def test_attention_mask_invariance(repo, attn_impl, human_name):
    """
    For each model:

      1) Single-sample invariance over all base token sets:
           no_padding, left_padding, right_padding,
           no_mask, mask_omitted.

      2) Batch without padding:
           stack all base token sets.

      3) Batch with padding:
           for each base, create right-padded + left-padded rows.

      All logits on the 4 real tokens must match single-sample "no_padding"
      for every base token set.
    """
    model_path = dllm.utils.resolve_with_base_env(repo, "BASE_MODELS_DIR")

    # Load model. We assume it's a decoder-style model with .logits.
    if attn_impl is None:
        model = transformers.AutoModel.from_pretrained(
            model_path,
            dtype=torch.float32,
            device_map="auto",
        ).eval()
    else:
        config = transformers.AutoConfig.from_pretrained(
            model_path,
            attn_implementation=attn_impl,
        )
        model = transformers.AutoModel.from_pretrained(
            model_path,
            config=config,
            dtype=torch.float32,
            device_map="auto",
        ).eval()

    # 1) Single-sample variants over all base token sets
    outs_single = _forward_variants(model)
    _assert_invariance(outs_single, human_name)
    single_no_pad = outs_single["no_padding"]  # [N,4,H]

    # 2) Batch (no padding)
    batch_nopad = _forward_batch_nopad(model)  # [N,4,H]

    # 3) Batch (padded, left+right)
    batch_padded = _forward_batch_padded(model)  # [2N,5,H]

    _assert_batch_equal_to_single(
        single_no_pad=single_no_pad,
        batch_nopad=batch_nopad,
        batch_padded=batch_padded,
        tag=human_name,
    )

    print(
        f"✅ {human_name} passed: mask invariance + batch (no-pad & padded) "
        f"consistency across {len(BASE_TOKEN_SETS)} base token sets within {ERROR_THRESHOLD}."
    )

    del model
    gc.collect()
    _cuda_cleanup()
