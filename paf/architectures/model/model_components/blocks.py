"""Blocks."""
from __future__ import annotations

from torch import nn

from .model_utils import init_weights

__all__ = ["block", "mid_blocks"]


def block(*, in_dim: int, out_dim: int) -> nn.Module:
    """Make a simple block."""
    linear = nn.Linear(in_dim, out_dim)
    relu = nn.GELU()
    b_norm = nn.BatchNorm1d(out_dim)
    seq = nn.Sequential(linear, relu, b_norm)
    seq.apply(init_weights)
    return seq


def mid_blocks(*, latent_dim: int, blocks: int) -> list[nn.Module]:
    """Build middle blocks for hidden layers."""
    return (
        [block(in_dim=latent_dim, out_dim=latent_dim) for _ in range(blocks - 1)]
        if blocks > 1
        else []
    )
