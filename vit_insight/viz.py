from __future__ import annotations

import io
from collections.abc import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from vit_insight.methods.rollout import attention_rollout, raw_attention  # noqa: E402
from vit_insight.registry import ModelSpec  # noqa: E402
from vit_insight.tokens import (  # noqa: E402
    cls_to_patches,
    patch_grid,
    patch_scores_to_map,
    upsample,
)


def _normalise(m: torch.Tensor) -> torch.Tensor:
    m = m - m.min()
    return m / m.max().clamp_min(1e-12)


def cls_heatmap(matrix: torch.Tensor, pixel_values: torch.Tensor, spec: ModelSpec) -> np.ndarray:
    """(S,S) attention-like matrix -> (H,W) heatmap in [0,1] over the input image."""
    scores = cls_to_patches(matrix, spec)
    grid = patch_grid(pixel_values, spec)
    m = patch_scores_to_map(scores, grid)
    return _normalise(upsample(m, tuple(pixel_values.shape[-2:]))).cpu().numpy()


def overlay(
    image: np.ndarray, heatmap: np.ndarray, alpha=0.5, cmap="jet", title=None
) -> Image.Image:
    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
    ax.imshow(image)
    ax.imshow(heatmap, cmap=cmap, alpha=alpha)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=12)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def rollout_overlay(result, spec, **kw) -> Image.Image:
    viz_kw = {k: kw.pop(k) for k in ("alpha", "cmap") if k in kw}
    r = attention_rollout(result.attentions, **kw)[0]
    return overlay(result.image, cls_heatmap(r, result.pixel_values, spec), **viz_kw)


def per_layer_frames(
    result, spec, layers: Sequence[int] | None = None, head_fusion="mean", alpha=0.5, cmap="jet"
) -> list[Image.Image]:
    layers = list(layers) if layers is not None else range(len(result.attentions))
    frames = []
    for i in layers:
        a = raw_attention(result.attentions, layer=i, head_fusion=head_fusion)[0]
        hm = cls_heatmap(a, result.pixel_values, spec)
        frames.append(overlay(result.image, hm, alpha, cmap, title=f"Layer {i + 1}"))
    return frames


def head_grid(result, spec, layer: int, alpha=0.5, cmap="jet") -> Image.Image:
    """One thumbnail per head of a layer."""
    attn = result.attentions[layer][0]  # (heads, S, S)
    n = attn.shape[0]
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.2 * rows), dpi=100)
    for h, ax in enumerate(np.array(axes).ravel()):
        ax.axis("off")
        if h < n:
            ax.imshow(result.image)
            ax.imshow(cls_heatmap(attn[h], result.pixel_values, spec), cmap=cmap, alpha=alpha)
            ax.set_title(f"L{layer + 1} H{h + 1}", fontsize=8)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def save_gif(frames: Sequence[Image.Image], path: str, duration_ms: int = 400) -> str:
    frames[0].save(
        path,
        format="GIF",
        save_all=True,
        append_images=list(frames[1:]),
        duration=duration_ms,
        loop=0,
    )
    return path
