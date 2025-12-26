import html
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.colorbar import ColorbarBase
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea, VPacker


# -----------------------------
# 1) Core: when does each position get unmasked?
# -----------------------------
def compute_reveal_steps_from_mask(
    token_steps: torch.Tensor,
    mask_token_ids: Union[int, Iterable[int]],
    *,
    mode: str = "first_unmask",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        token_steps: Long tensor (S,B,L)
        mask_token_ids: int or iterable[int] for mask token id(s)
        mode:
            - "first_unmask": first step where token != mask (best for "mask -> unmask" demo)
            - "last_change": last step where token changes (useful for iterative refinement)
    Returns:
        reveal_steps: Long tensor (B,L)
            - first_unmask: in [0..S-1], or S if never unmasked
            - last_change:  in [0..S-1]
        final_ids: Long tensor (B,L) = token_steps[-1]
    """
    if token_steps.ndim != 3:
        raise ValueError(f"token_steps must have shape (S,B,L), got {tuple(token_steps.shape)}")
    S, B, L = token_steps.shape

    if isinstance(mask_token_ids, int):
        mask_ids = torch.tensor([mask_token_ids], device=token_steps.device, dtype=token_steps.dtype)
    else:
        mask_ids_list = list(mask_token_ids)
        if len(mask_ids_list) == 0:
            raise ValueError("mask_token_ids cannot be empty")
        mask_ids = torch.tensor(mask_ids_list, device=token_steps.device, dtype=token_steps.dtype)

    final_ids = token_steps[-1]

    if mode == "first_unmask":
        is_mask = (token_steps.unsqueeze(-1) == mask_ids).any(dim=-1)  # (S,B,L)
        non_mask = ~is_mask

        idx = torch.arange(S, device=token_steps.device, dtype=torch.long).view(S, 1, 1)
        first = torch.where(non_mask, idx, torch.full_like(idx, S))
        reveal = first.min(dim=0).values  # (B,L), S means never unmasked
        return reveal, final_ids

    if mode == "last_change":
        diff = token_steps[1:] != token_steps[:-1]  # (S-1,B,L)
        idx = torch.arange(1, S, device=token_steps.device, dtype=torch.long).view(S - 1, 1, 1)
        last = torch.where(diff, idx, torch.zeros_like(idx))
        reveal = last.max(dim=0).values
        return reveal, final_ids

    raise ValueError(f"Unknown mode: {mode}")


# -----------------------------
# 2) Coloring: dark blue -> light blue
# -----------------------------
def default_reveal_cmap() -> LinearSegmentedColormap:
    # Dark blue -> very light blue (instead of black -> cyan)
    return LinearSegmentedColormap.from_list("reveal_blue", ["#081a3a", "#b9e6ff"])


def steps_to_rgba(
    reveal_steps_1d: torch.Tensor,
    num_steps: int,
    *,
    cmap=None,
    never_color=(0.60, 0.60, 0.60, 1.0),
) -> np.ndarray:
    if cmap is None:
        cmap = default_reveal_cmap()

    rs = reveal_steps_1d.detach().cpu().numpy().astype(np.int64)
    denom = max(1, num_steps - 1)
    t = np.clip(rs, 0, num_steps - 1) / denom  # 0..1
    rgba = np.array(cmap(t), copy=True)

    # In first_unmask mode, never-unmasked positions have reveal step == num_steps
    rgba[rs >= num_steps] = never_color
    return rgba


# -----------------------------
# 3) Token -> per-token pieces (for per-token coloring)
# -----------------------------
def decode_token_pieces(
    token_ids_1d: Union[torch.Tensor, List[int], np.ndarray],
    *,
    tokenizer=None,
    decode_fn: Optional[Callable[[int], str]] = None,
    skip_special_tokens: bool = False,
    clean_up_tokenization_spaces: bool = False,
) -> List[str]:
    """
    Provide either:
      - tokenizer with .decode (HF tokenizer), OR
      - decode_fn: (int)->str
    """
    if decode_fn is None:
        if tokenizer is None or not hasattr(tokenizer, "decode"):
            raise ValueError("Provide either tokenizer (with .decode) or decode_fn.")

        def decode_fn(x: int) -> str:
            return tokenizer.decode(
                [int(x)],
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            )

    if isinstance(token_ids_1d, torch.Tensor):
        ids_list = token_ids_1d.detach().cpu().tolist()
    elif isinstance(token_ids_1d, np.ndarray):
        ids_list = token_ids_1d.tolist()
    else:
        ids_list = list(token_ids_1d)

    cache: Dict[int, str] = {}
    out: List[str] = []
    for tid in ids_list:
        tid_int = int(tid)
        if tid_int not in cache:
            cache[tid_int] = decode_fn(tid_int)
        out.append(cache[tid_int])
    return out


# -----------------------------
# 4) Matplotlib text box that FITS the content
#    - auto-expands figure if needed
#    - border drawn around actual rendered text extent
# -----------------------------
def _build_text_vbox(
    pieces: List[str],
    rgba: np.ndarray,
    *,
    fontfamily: str,
    fontsize: int,
    line_sep: int,
) -> VPacker:
    line_children = []
    current = []

    def flush_line():
        nonlocal current
        if current:
            line_children.append(HPacker(children=current, align="baseline", pad=0, sep=0))
        else:
            # preserve empty line height
            line_children.append(
                TextArea(
                    " ",
                    textprops={"color": (0, 0, 0, 0), "fontfamily": fontfamily, "fontsize": fontsize},
                )
            )
        current = []

    for piece, c in zip(pieces, rgba):
        if piece == "":
            continue
        piece = piece.replace("\t", "    ")
        parts = piece.split("\n")
        for i, part in enumerate(parts):
            if part != "":
                current.append(TextArea(part, textprops={"color": c, "fontfamily": fontfamily, "fontsize": fontsize}))
            if i < len(parts) - 1:
                flush_line()

    flush_line()
    return VPacker(children=line_children, align="left", pad=0, sep=line_sep)


def _add_colored_text(
    ax,
    pieces: List[str],
    rgba: np.ndarray,
    *,
    fontfamily: str,
    fontsize: int,
    line_sep: int,
) -> AnchoredOffsetbox:
    vbox = _build_text_vbox(pieces, rgba, fontfamily=fontfamily, fontsize=fontsize, line_sep=line_sep)
    ax.set_axis_off()

    anchored = AnchoredOffsetbox(
        loc="upper left",
        child=vbox,
        pad=0.0,
        frameon=False,
        bbox_to_anchor=(0.0, 1.0),
        bbox_transform=ax.transAxes,
        borderpad=0.0,
    )
    anchored.set_clip_on(False)  # IMPORTANT: don't clip if it grows beyond axes
    ax.add_artist(anchored)
    return anchored


def _auto_expand_figure_to_fit(
    ax,
    artist,
    *,
    pad_px: int = 14,
    max_iters: int = 4,
    max_width_in: Optional[float] = 28.0,
    max_height_in: Optional[float] = 18.0,
):
    """
    Expand the figure size until `artist` fits inside the axes area (plus padding).
    This prevents the text from being clipped / spilling outside the box.
    """
    fig = ax.figure
    for _ in range(max_iters):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        artist_bb = artist.get_window_extent(renderer=renderer)
        ax_bb = ax.get_window_extent(renderer=renderer)

        scale_w = (artist_bb.width + 2 * pad_px) / max(1.0, ax_bb.width)
        scale_h = (artist_bb.height + 2 * pad_px) / max(1.0, ax_bb.height)

        if scale_w <= 1.01 and scale_h <= 1.01:
            break

        w, h = fig.get_size_inches()
        new_w = w * max(1.0, scale_w)
        new_h = h * max(1.0, scale_h)

        if max_width_in is not None:
            new_w = min(new_w, max_width_in)
        if max_height_in is not None:
            new_h = min(new_h, max_height_in)

        # If clamped and still doesn't fit, we stop trying (avoid infinite loop).
        if (max_width_in is not None and abs(new_w - w) < 1e-6) and (max_height_in is not None and abs(new_h - h) < 1e-6):
            break

        fig.set_size_inches(new_w, new_h, forward=True)


def _draw_border_around_artist(
    ax,
    artist,
    *,
    pad_px: int = 10,
    lw: float = 1.0,
):
    """
    Draw the border around the artist's TRUE rendered extent (not just axes [0,1] box).
    This fixes "text exceeds bounding box".
    """
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    bb = artist.get_window_extent(renderer=renderer)
    x0, y0, x1, y1 = bb.x0 - pad_px, bb.y0 - pad_px, bb.x1 + pad_px, bb.y1 + pad_px

    inv = ax.transAxes.inverted()
    (ax_x0, ax_y0) = inv.transform((x0, y0))
    (ax_x1, ax_y1) = inv.transform((x1, y1))

    rect = plt.Rectangle(
        (ax_x0, ax_y0),
        ax_x1 - ax_x0,
        ax_y1 - ax_y0,
        transform=ax.transAxes,
        fill=False,
        lw=lw,
        edgecolor="black",
        clip_on=False,  # IMPORTANT: allow border outside axes if needed
    )
    ax.add_patch(rect)


def render_generation_matplotlib(
    token_steps: torch.Tensor,
    mask_token_ids: Union[int, Iterable[int]],
    *,
    batch_idx: int = 0,
    tokenizer=None,
    decode_fn: Optional[Callable[[int], str]] = None,
    mode: str = "first_unmask",
    figsize: Tuple[float, float] = (14, 3.2),
    fontfamily: str = "DejaVu Sans Mono",
    fontsize: int = 18,
    cmap=None,
    never_color=(0.6, 0.6, 0.6, 1.0),
    legend_title: str = "Generated tokens:",
    legend_style: str = "token_bar",  # "token_bar" or "colorbar"
    # NEW:
    auto_expand: bool = True,
    border_pad_px: int = 10,
    expand_pad_px: int = 14,
) -> plt.Figure:
    if cmap is None:
        cmap = default_reveal_cmap()

    reveal, final_ids = compute_reveal_steps_from_mask(token_steps, mask_token_ids, mode=mode)
    S, B, L = token_steps.shape

    reveal_b = reveal[batch_idx]
    ids_b = final_ids[batch_idx]

    pieces = decode_token_pieces(ids_b, tokenizer=tokenizer, decode_fn=decode_fn)
    rgba = steps_to_rgba(reveal_b, num_steps=S, cmap=cmap, never_color=never_color)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[0.28, 1.0], hspace=0.06)

    # --- Legend row
    ax_top = fig.add_subplot(gs[0])
    ax_top.set_axis_off()
    top_pos = ax_top.get_position()

    cb_ax = fig.add_axes(
        [
            top_pos.x0 + 0.44 * top_pos.width,   # x
            top_pos.y0 + 0.25 * top_pos.height,  # y
            0.52 * top_pos.width,                # w
            0.50 * top_pos.height,               # h
        ]
    )

    if legend_style == "colorbar":
        cb = ColorbarBase(cb_ax, cmap=cmap, norm=Normalize(0, 1), orientation="horizontal")
        cb.set_ticks([])
    elif legend_style == "token_bar":
        denom = max(1, S - 1)
        t = (torch.clamp(reveal_b, 0, S - 1).float() / denom).detach().cpu().numpy()
        never_mask = (reveal_b.detach().cpu().numpy() >= S)
        t_ma = np.ma.array(t, mask=never_mask)

        cmap2 = cmap
        try:
            cmap2 = cmap.copy()
        except Exception:
            pass
        try:
            cmap2.set_bad(color=never_color)
        except Exception:
            pass

        cb_ax.imshow(t_ma[None, :], aspect="auto", cmap=cmap2, vmin=0, vmax=1, interpolation="nearest")
        cb_ax.set_xticks([])
        cb_ax.set_yticks([])
    else:
        raise ValueError("legend_style must be 'token_bar' or 'colorbar'")

    for spine in cb_ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)

    fig.text(
        top_pos.x0,
        top_pos.y0 + 0.5 * top_pos.height,
        legend_title,
        ha="left",
        va="center",
        fontsize=fontsize + 2,
        fontfamily="serif",
    )
    cb_ax.text(-0.04, 0.5, "t = 0", transform=cb_ax.transAxes, ha="right", va="center", fontsize=fontsize, fontfamily="serif")
    cb_ax.text(1.04, 0.5, "t = 1", transform=cb_ax.transAxes, ha="left", va="center", fontsize=fontsize, fontfamily="serif")

    ax_top.plot([0, 1], [0.05, 0.05], transform=ax_top.transAxes, color="black", lw=1)

    # --- Text row
    ax_txt = fig.add_subplot(gs[1])
    anchored = _add_colored_text(ax_txt, pieces, rgba, fontfamily=fontfamily, fontsize=fontsize, line_sep=2)

    # NEW: auto-expand so nothing is clipped / overflowing
    if auto_expand:
        _auto_expand_figure_to_fit(ax_txt, anchored, pad_px=expand_pad_px)

    # NEW: border around TRUE text extent (fixes "text exceeds bounding box")
    _draw_border_around_artist(ax_txt, anchored, pad_px=border_pad_px, lw=1.0)

    return fig


# -----------------------------
# 5) HTML renderer (optional) with matching blue gradient
# -----------------------------
def render_generation_html(
    token_steps: torch.Tensor,
    mask_token_ids: Union[int, Iterable[int]],
    *,
    batch_idx: int = 0,
    tokenizer=None,
    decode_fn: Optional[Callable[[int], str]] = None,
    mode: str = "first_unmask",
    font_family: str = "Menlo,Consolas,Monaco,'DejaVu Sans Mono',monospace",
    font_size_px: int = 22,
    line_height: float = 1.35,
    show_legend: bool = True,
    legend_title: str = "Generated tokens:",
    never_color_css: str = "#999999",
    return_html_str: bool = False,
):
    cmap = default_reveal_cmap()

    reveal, final_ids = compute_reveal_steps_from_mask(token_steps, mask_token_ids, mode=mode)
    S = token_steps.shape[0]
    reveal_b = reveal[batch_idx]
    ids_b = final_ids[batch_idx]

    pieces = decode_token_pieces(ids_b, tokenizer=tokenizer, decode_fn=decode_fn)
    rgba = steps_to_rgba(reveal_b, num_steps=S, cmap=cmap)

    def rgba_to_css(c) -> str:
        r, g, b, a = c
        return f"rgba({int(r*255)},{int(g*255)},{int(b*255)},{a:.3f})"

    spans: List[str] = []
    reveal_list = reveal_b.detach().cpu().tolist()
    for piece, c, st in zip(pieces, rgba, reveal_list):
        if piece == "":
            continue
        piece = piece.replace("\t", "    ")
        safe = html.escape(piece)
        color_css = never_color_css if st >= S else rgba_to_css(c)
        spans.append(f'<span style="color:{color_css};">{safe}</span>')

    legend_html = ""
    if show_legend:
        # Match the new colormap: dark blue -> light blue
        legend_html = f"""
        <div style="display:flex; align-items:center; gap:16px; margin-bottom:8px;">
          <div style="font-family:serif; font-size:{font_size_px+4}px;">{html.escape(legend_title)}</div>
          <div style="font-family:serif; font-size:{font_size_px}px;">t = 0</div>
          <div style="height:14px; width:420px; border:1px solid #000;
                      background: linear-gradient(to right, #081a3a, #b9e6ff);"></div>
          <div style="font-family:serif; font-size:{font_size_px}px;">t = 1</div>
        </div>
        <hr style="border:0; border-top:1px solid #000; margin: 6px 0 10px 0;">
        """

    body_html = f"""
    <div style="max-width: 1600px;">
      {legend_html}
      <pre style="margin:0; white-space:pre; overflow-x:auto;
                  font-family:{font_family}; font-size:{font_size_px}px; line-height:{line_height};">
{''.join(spans)}
      </pre>
    </div>
    """

    if return_html_str:
        return body_html

    try:
        from IPython.display import HTML  # type: ignore
        return HTML(body_html)
    except Exception:
        return body_html


# -----------------------------
# Usage examples
# -----------------------------
if __name__ == "__main__":
    # Example with a HuggingFace tokenizer:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("/home/lingjie7/models/huggingface/GSAI-ML/LLaDA-8B-Instruct/")
    mask_token = "<|mdm_mask|>"
    mask_id = 126336
    print(mask_id)
    token_steps = torch.load('test_histories.pt')
    token_steps = torch.stack(token_steps, dim=0)  # (S,B,L)
    print(token_steps.shape)
    fig = render_generation_matplotlib(token_steps, mask_id, tokenizer=tokenizer, batch_idx=0)
    fig.savefig("generation_demo.png", dpi=200, bbox_inches="tight")

    pass


