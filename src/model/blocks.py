"""Blocks."""
from typing import List

from torch import nn

from src.model.model_utils import init_weights


def block(*, in_dim: int, out_dim: int) -> nn.Module:
    """Make a simple block."""
    linear = nn.Linear(in_dim, out_dim)
    relu = nn.SELU()
    seq = nn.Sequential(linear, relu)
    seq.apply(init_weights)
    return seq


def mid_blocks(*, latent_dim: int, blocks: int) -> List:
    """Build middle blocks for hidden layers."""
    return [block(in_dim=latent_dim, out_dim=latent_dim) for _ in range(blocks - 1)] if blocks > 1 else []
