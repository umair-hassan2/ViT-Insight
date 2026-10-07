from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from vit_insight.loading import LoadedModel


@dataclass
class ForwardResult:
    image: np.ndarray  # (H, W, 3) in [0,1], the model's actual input, de-normalised
    pixel_values: torch.Tensor  # (1, C, H, W)
    attentions: tuple[torch.Tensor, ...]  # per layer (1, heads, S, S)
    logits: torch.Tensor | None  # (1, C) or (1, T); None for headless encoders
    pred: int | None

    @property
    def pred_prob(self) -> float | None:
        if self.logits is None:
            return None
        return torch.softmax(self.logits, dim=-1)[0, self.pred].item()


def _denormalise(pixel_values: torch.Tensor, image_processor) -> np.ndarray:
    mean = torch.tensor(image_processor.image_mean).view(-1, 1, 1)
    std = torch.tensor(image_processor.image_std).view(-1, 1, 1)
    img = pixel_values[0].detach().cpu() * std + mean
    return img.clamp(0, 1).permute(1, 2, 0).numpy()


def _extract_attentions(outputs) -> tuple[torch.Tensor, ...]:
    vis = getattr(outputs, "vision_model_output", None)
    attns = vis.attentions if vis is not None else outputs.attentions
    if attns is None or attns[0] is None:
        raise RuntimeError(
            "attentions are None; model must be loaded with attn_implementation='eager'"
        )
    return tuple(attns)


@torch.no_grad()
def run_forward(
    lm: LoadedModel, image: Image.Image, text_labels: Sequence[str] | None = None
) -> ForwardResult:
    spec = lm.spec
    if spec.text_conditioned:
        if not text_labels:
            raise ValueError(f"{spec.id} needs text labels")
        inputs = lm.processor(
            text=list(text_labels), images=image, return_tensors="pt", padding=True
        )
    else:
        inputs = lm.processor(images=image, return_tensors="pt")
    inputs = {k: v.to(lm.device) for k, v in inputs.items()}
    outputs = lm.model(**inputs, output_attentions=True)

    logits = getattr(outputs, "logits_per_image", None)
    if logits is None:
        logits = getattr(outputs, "logits", None)
    pred = int(logits.argmax(dim=-1).item()) if logits is not None else None

    pv = inputs["pixel_values"]
    return ForwardResult(
        image=_denormalise(pv, lm.image_processor),
        pixel_values=pv,
        attentions=_extract_attentions(outputs),
        logits=logits.detach().cpu() if logits is not None else None,
        pred=pred,
    )
