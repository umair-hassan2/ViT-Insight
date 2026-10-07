from __future__ import annotations

import torch

from vit_insight.registry import ModelSpec


def patch_grid(pixel_values: torch.Tensor, spec: ModelSpec) -> tuple[int, int]:
    """(rows, cols) of the patch grid for a (B,C,H,W) input."""
    h, w = pixel_values.shape[-2:]
    return h // spec.patch_size, w // spec.patch_size


def patch_scores_to_map(scores: torch.Tensor, grid: tuple[int, int]) -> torch.Tensor:
    """(N,) or (B,N) patch scores -> (rows, cols) or (B, rows, cols)."""
    rows, cols = grid
    if scores.shape[-1] != rows * cols:
        raise ValueError(f"{scores.shape[-1]} patch scores do not fit grid {grid}")
    return scores.reshape(*scores.shape[:-1], rows, cols)


def cls_to_patches(matrix: torch.Tensor, spec: ModelSpec) -> torch.Tensor:
    """Row of the CLS query over patch keys. matrix: (S,S) or (B,S,S)."""
    if not spec.cls_attention:
        raise ValueError(f"{spec.id} has no CLS token; CLS-row methods are unavailable")
    return matrix[..., 0, spec.prefix_tokens :]


def upsample(map2d: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """(rows, cols) -> (H, W) bilinear."""
    x = map2d[None, None].float()
    return torch.nn.functional.interpolate(x, size=size, mode="bilinear", align_corners=False)[0, 0]
