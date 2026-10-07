import pytest
import torch


@pytest.fixture
def two_layer_attn():
    """Two deterministic 3-token single-head attention matrices (batch 1)."""
    a1 = torch.tensor([[[[0.6, 0.3, 0.1], [0.2, 0.7, 0.1], [0.1, 0.1, 0.8]]]])
    a2 = torch.tensor([[[[0.1, 0.8, 0.1], [0.3, 0.3, 0.4], [0.5, 0.25, 0.25]]]])
    return [a1, a2]
