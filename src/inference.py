import torch.nn.functional as F
import torch

import matplotlib.pyplot as plt
from PIL import Image
import io
import numpy as np
import tempfile
import os


def attention_rollout(attentions, start_layer=0):
    device = attentions[0].device
    rollout = torch.eye(attentions[0].size(-1), device=device)
    for attn in attentions[start_layer:]:
        attn_avg = attn.mean(dim=1)
        attn_avg = attn_avg / (attn_avg.sum(dim=-1, keepdim=True) + 1e-6)
        rollout = rollout @ attn_avg
    return rollout


def _save_fig_to_temp(fig, suffix):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        fig.savefig(tmp.name, bbox_inches='tight', pad_inches=0)
    finally:
        plt.close(fig)
    return tmp.name


def generate_attention_frames(img, attentions, mode="Per-layer GIF", alpha=0.5, colormap="jet", layer_range=None):
    frames = []
    num_layers = len(attentions)

    if layer_range is None:
        layer_range = (0, num_layers)
    start, end = layer_range
    start = max(0, int(start))
    end = min(num_layers, int(end) if end is not None else num_layers)
    if start >= end:
        start, end = 0, num_layers

    H_img, W_img = img.shape[:2]

    if mode == "Attention Rollout":
        rollout = attention_rollout(attentions[start:end])  # (B, S, S)
        rollout0 = rollout[0] if rollout.dim() == 3 else rollout  # (S, S)
        cls_rollout = rollout0[0, 1:]  # (S-1)
        num_patches = int(cls_rollout.numel() ** 0.5)
        rollout_map = cls_rollout.reshape(1, num_patches, num_patches)  # (1, h, w)
        rollout_map = F.interpolate(rollout_map.unsqueeze(0), size=(H_img, W_img), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img)
        ax.imshow(rollout_map, cmap=colormap, alpha=alpha)
        ax.axis("off")
        return _save_fig_to_temp(fig, suffix=".png")

    # Per-layer GIF
    for l in range(start, end):
        attn = attentions[l]  # (B, H, S, S)
        attn_avg = attn.mean(dim=1)[0]  # (S, S) -> use first image in batch
        cls_to_patches = attn_avg[0, 1:]  # (S-1)
        num_patches = int(cls_to_patches.numel() ** 0.5)
        attn_map = cls_to_patches.reshape(1, num_patches, num_patches)  # (1, h, w)
        heatmap = F.interpolate(attn_map.unsqueeze(0), size=(H_img, W_img), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img, alpha=0.7)
        ax.imshow(heatmap, cmap=colormap, alpha=alpha)
        ax.axis("off")
        ax.set_title(f"Layer {l+1}", fontsize=14, color='white', backgroundcolor='black', alpha=0.5)

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        # Use RGB to avoid dependency on Image.ADAPTIVE constant across Pillow versions
        frames.append(Image.open(buf).convert('RGB'))

    # Save animated GIF to a temporary file and return its path
    tmp_gif = tempfile.NamedTemporaryFile(delete=False, suffix=".gif")
    # Ensure at least one frame exists
    if len(frames) == 1:
        frames[0].save(tmp_gif.name, format='GIF', save_all=True, duration=400)
    else:
        frames[0].save(tmp_gif.name, format='GIF', append_images=frames[1:], save_all=True, duration=400)
    return tmp_gif.name