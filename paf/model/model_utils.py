"""Model related utiltiy functions."""
from __future__ import annotations

import torch
from torch import Tensor, arange, autograd, nn, stack
import torch.nn.functional as F


def init_weights(m: nn.Module) -> None:
    """Make Linear layer weights initialised with Xavier Norm."""
    # if type(m) == nn.Linear:
    # nn.init.xavier_uniform_(m.weight)


def index_by_s(recons: list[Tensor], s: Tensor) -> Tensor:
    """Get recon by the index of S."""
    _recons = stack(recons, dim=1)
    return _recons[arange(_recons.shape[0]), s.long()]


def augment_recons(x: Tensor, cf_x: Tensor, s: Tensor) -> list[Tensor]:
    """Given real data and counterfactuial data, return in recon format based on S index."""
    dd = [(x[i], cf_x[i]) if _s == 0 else (cf_x[i], x[i]) for i, _s in enumerate(s)]
    return list((torch.stack([d[0] for d in dd]), torch.stack([d[1] for d in dd])))


def to_discrete(*, inputs: Tensor) -> Tensor:
    """Discretize the data."""
    if inputs.dim() <= 1 or inputs.size(1) <= 1:
        return inputs.round()
    argmax = inputs.argmax(dim=1)
    return F.one_hot(argmax, num_classes=inputs.size(1))


class GradReverse(autograd.Function):
    """Gradient reversal layer."""

    @staticmethod
    def forward(ctx: autograd.Function, x: Tensor, lambda_: float) -> Tensor:
        """Do GRL."""
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx: autograd.Function, grad_output: Tensor) -> tuple[Tensor, Tensor | None]:
        """Do GRL."""
        return grad_output.neg().mul(ctx.lambda_), None


def grad_reverse(features: Tensor, lambda_: float = 1.0) -> Tensor:
    """Gradient Reversal layer."""
    return GradReverse.apply(features, lambda_)
