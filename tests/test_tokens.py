import pytest
import torch

from vit_insight.registry import ModelSpec, Registry
from vit_insight.tokens import cls_to_patches, patch_grid, patch_scores_to_map


def _spec(prefix, patch=16, cls=True):
    return ModelSpec("x", "x", "vit", "sup", prefix, patch, 12, cls_attention=cls)


@pytest.mark.parametrize("prefix", [1, 2, 5])
def test_cls_to_patches_strips_prefix(prefix):
    s = 196 + prefix
    m = torch.rand(s, s)
    out = cls_to_patches(m, _spec(prefix))
    assert out.shape == (196,)
    assert torch.equal(out, m[0, prefix:])


def test_grid_from_pixel_values_not_sqrt():
    pv = torch.zeros(1, 3, 224, 224)
    assert patch_grid(pv, _spec(1, 14)) == (16, 16)
    assert patch_grid(pv, _spec(1, 32)) == (7, 7)
    assert patch_scores_to_map(torch.zeros(256), (16, 16)).shape == (16, 16)


def test_no_cls_model_rejects_cls_methods():
    with pytest.raises(ValueError):
        cls_to_patches(torch.rand(4, 4), _spec(0, cls=False))


def test_registry_loads_and_is_consistent():
    reg = Registry()
    assert "google/vit-base-patch16-224" in reg
    assert reg.get("facebook/dinov2-with-registers-base").prefix_tokens == 5
    assert reg.get("facebook/deit-base-distilled-patch16-224").prefix_tokens == 2
    assert reg.get("openai/clip-vit-base-patch16").text_conditioned
    assert not reg.get("google/siglip-base-patch16-224").cls_attention
