"""End-to-end on a randomly initialised 2-layer ViT: no downloads, runs on CPU in seconds."""

import numpy as np
import pytest
import torch
from PIL import Image

from vit_insight.inference import run_forward
from vit_insight.loading import LoadedModel
from vit_insight.registry import ModelSpec
from vit_insight.viz import head_grid, per_layer_frames, rollout_overlay

transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def tiny():
    from transformers import ViTConfig, ViTForImageClassification, ViTImageProcessor

    cfg = ViTConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        image_size=32,
        patch_size=8,
        num_labels=3,
        attn_implementation="eager",
    )
    torch.manual_seed(0)
    model = ViTForImageClassification(cfg).eval()
    proc = ViTImageProcessor(size={"height": 32, "width": 32})
    spec = ModelSpec("tiny", "tiny", "vit", "random", 1, 8, 2)
    return LoadedModel(spec, proc, model, torch.device("cpu"))


def test_forward_shapes_and_denorm(tiny):
    img = Image.fromarray(np.random.randint(0, 255, (48, 64, 3), dtype=np.uint8))
    res = run_forward(tiny, img)
    assert len(res.attentions) == 2
    assert res.attentions[0].shape == (1, 2, 17, 17)  # 1 CLS + 16 patches
    assert res.image.shape == (32, 32, 3) and 0 <= res.image.min() and res.image.max() <= 1
    assert res.pred in {0, 1, 2}


def test_visualisations_render(tiny):
    img = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
    res = run_forward(tiny, img)
    assert rollout_overlay(res, tiny.spec, discard_ratio=0.5).size[0] > 0
    assert len(per_layer_frames(res, tiny.spec)) == 2
    assert head_grid(res, tiny.spec, layer=1).size[0] > 0
