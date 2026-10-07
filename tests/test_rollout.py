import torch

from vit_insight.methods.rollout import attention_rollout, fuse_heads, legacy_rollout, raw_attention


def _tilde(a):
    eye = torch.eye(a.shape[-1])
    t = 0.5 * a + 0.5 * eye
    return t / t.sum(-1, keepdim=True)


def test_rollout_matches_hand_computation(two_layer_attn):
    a1, a2 = (x[0, 0] for x in two_layer_attn)
    expected = _tilde(a2) @ _tilde(a1)  # later layer on the left
    got = attention_rollout(two_layer_attn)[0]
    assert torch.allclose(got, expected, atol=1e-6)


def test_rollout_order_is_not_commutative(two_layer_attn):
    a1, a2 = (x[0, 0] for x in two_layer_attn)
    wrong = _tilde(a1) @ _tilde(a2)
    got = attention_rollout(two_layer_attn)[0]
    assert not torch.allclose(got, wrong, atol=1e-4)


def test_rollout_rows_are_stochastic(two_layer_attn):
    r = attention_rollout(two_layer_attn, discard_ratio=0.3)
    assert torch.allclose(r.sum(-1), torch.ones(1, 3), atol=1e-6)


def test_rollout_without_residual(two_layer_attn):
    a1, a2 = (x[0, 0] for x in two_layer_attn)
    got = attention_rollout(two_layer_attn, residual=False)[0]
    assert torch.allclose(got, a2 @ a1, atol=1e-6)


def test_legacy_differs_from_corrected(two_layer_attn):
    assert not torch.allclose(
        legacy_rollout(two_layer_attn), attention_rollout(two_layer_attn), atol=1e-3
    )


def test_layer_range(two_layer_attn):
    a1 = two_layer_attn[0][0, 0]
    got = attention_rollout(two_layer_attn, start_layer=0, end_layer=1)[0]
    assert torch.allclose(got, _tilde(a1), atol=1e-6)


def test_head_fusion_and_raw():
    attn = torch.rand(1, 4, 5, 5)
    assert torch.allclose(fuse_heads(attn, "max"), attn.max(1).values)
    assert torch.allclose(raw_attention([attn], layer=0), attn.mean(1))


def test_discard_never_zeroes_cls_column():
    attn = torch.rand(1, 1, 6, 6)
    attn[..., 0] = 1e-9  # make CLS column the smallest everywhere
    r = attention_rollout([attn], discard_ratio=0.5, residual=False)
    assert (r[0, :, 0] > 0).all()
