"""Attention rollout (Abnar & Zuidema, ACL 2020) and raw-attention baselines.

attentions: sequence of per-layer tensors shaped (B, heads, S, S), layer 0 first.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


def fuse_heads(attn: torch.Tensor, how: str = "mean") -> torch.Tensor:
    if how == "mean":
        return attn.mean(dim=1)
    if how == "max":
        return attn.max(dim=1).values
    if how == "min":
        return attn.min(dim=1).values
    raise ValueError(f"unknown head fusion {how!r}")


def _discard_lowest(a: torch.Tensor, ratio: float, keep_col: int | None = 0) -> torch.Tensor:
    """Zero the lowest `ratio` fraction of entries per row (never the CLS column)."""
    if ratio <= 0:
        return a
    a = a.clone()
    k = int(a.shape[-1] * ratio)
    if k == 0:
        return a
    idx = a.topk(k, dim=-1, largest=False).indices
    mask = torch.zeros_like(a, dtype=torch.bool).scatter_(-1, idx, True)
    if keep_col is not None:
        mask[..., keep_col] = False
    return a.masked_fill(mask, 0.0)


def attention_rollout(
    attentions: Sequence[torch.Tensor],
    head_fusion: str = "mean",
    discard_ratio: float = 0.0,
    residual: bool = True,
    start_layer: int = 0,
    end_layer: int | None = None,
) -> torch.Tensor:
    """Rollout Ã = Ã_L @ ... @ Ã_1 with Ã_l = rownorm(½A_l + ½I). Returns (B, S, S)."""
    layers = attentions[start_layer:end_layer]
    if not layers:
        raise ValueError("empty layer range")
    b, _, s, _ = layers[0].shape
    eye = torch.eye(s, device=layers[0].device, dtype=layers[0].dtype)
    result = eye.expand(b, s, s).clone()
    for attn in layers:
        a = fuse_heads(attn, head_fusion)
        a = _discard_lowest(a, discard_ratio)
        if residual:
            a = 0.5 * a + 0.5 * eye
        a = a / a.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        # current layer on the left: later layers compose on top of earlier ones
        result = a @ result
    return result


def legacy_rollout(attentions: Sequence[torch.Tensor]) -> torch.Tensor:
    """The pre-0.2 implementation: no residual term, reversed product. Kept for ablation."""
    s = attentions[0].size(-1)
    rollout = torch.eye(s, device=attentions[0].device)
    for attn in attentions:
        a = attn.mean(dim=1)
        a = a / (a.sum(dim=-1, keepdim=True) + 1e-6)
        rollout = rollout @ a
    return rollout


def raw_attention(
    attentions: Sequence[torch.Tensor], layer: int = -1, head_fusion: str = "mean"
) -> torch.Tensor:
    """Head-fused attention of a single layer. Returns (B, S, S)."""
    return fuse_heads(attentions[layer], head_fusion)
