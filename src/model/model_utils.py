"""Model related utiltiy functions."""
from typing import List, Optional, Tuple

from torch import Tensor, autograd, cat, gather, nn


def init_weights(m: nn.Module) -> None:
    """Make Linear layer weights initialised with Xavier Norm."""
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


def index_by_s(recons: List[Tensor], s: Tensor) -> Tensor:
    """Get recon by the index of S."""
    _recons = cat(recons, dim=1)
    return gather(_recons, 1, s.unsqueeze(-1).long())


class GradReverse(autograd.Function):
    """Gradient reversal layer."""

    @staticmethod
    def forward(ctx: autograd.Function, x: Tensor, lambda_: float) -> Tensor:
        """Do GRL."""
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx: autograd.Function, grad_output: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """Do GRL."""
        return grad_output.neg().mul(ctx.lambda_), None


def grad_reverse(features: Tensor, lambda_: float = 1.0) -> Tensor:
    """Gradient Reversal layer."""
    return GradReverse.apply(features, lambda_)
