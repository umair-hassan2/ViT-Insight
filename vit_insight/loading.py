from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import torch

from vit_insight.registry import ModelSpec, Registry

_registry = Registry()


def pick_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class LoadedModel:
    spec: ModelSpec
    processor: Any
    model: torch.nn.Module
    device: torch.device

    @property
    def image_processor(self):
        # CLIP/SigLIP processors wrap an image processor; ViT processors are one already.
        return getattr(self.processor, "image_processor", self.processor)


def _model_class(spec: ModelSpec):
    from transformers import AutoModel, AutoModelForImageClassification, CLIPModel, SiglipModel

    if spec.family == "clip":
        return CLIPModel
    if spec.family == "siglip":
        return SiglipModel
    if spec.family in {"vit", "deit"}:
        return AutoModelForImageClassification
    return AutoModel


@lru_cache(maxsize=4)
def _load_cached(model_id: str, device_str: str) -> LoadedModel:
    from transformers import AutoProcessor

    spec = _registry.get(model_id)
    processor = AutoProcessor.from_pretrained(model_id)
    # SDPA/flash kernels return attentions=None; eager is required for output_attentions.
    model = _model_class(spec).from_pretrained(model_id, attn_implementation="eager")
    device = torch.device(device_str)
    model.eval().to(device)
    return LoadedModel(spec=spec, processor=processor, model=model, device=device)


def load_model(model_id: str, device: str | None = None) -> LoadedModel:
    return _load_cached(model_id, str(pick_device(device)))
