"""MMD functions."""
from __future__ import annotations
import logging
from typing import Any, Sequence, NamedTuple

import torch
from torch import Tensor

__all__ = ["mmd2"]

from paf.config_classes.dataclasses import KernelType

log = logging.getLogger(__name__)


class KernelOut(NamedTuple):
    xx: Tensor
    xy: Tensor
    yy: Tensor
    const_diag: float


def _dot_kernel(x: Tensor, y: Tensor) -> KernelOut:
    xx_gm = x @ x.t()
    xy_gm = x @ y.t()
    yy_gm = y @ y.t()

    return KernelOut(xx=xx_gm, xy=xy_gm, yy=yy_gm, const_diag=0.0)


def _mix_rq_kernel(
    x: Tensor,
    y: Tensor,
    scales: Sequence[float] | None = None,
    wts: Sequence[float] | None = None,
    add_dot: float = 0.0,
) -> KernelOut:
    """Rational quadratic kernel.

    http://www.cs.toronto.edu/~duvenaud/cookbook/index.html
    """
    scales = (0.1, 1.0, 10.0) if scales is None else scales
    wts = [1.0] * len(scales) if wts is None else wts

    xx_gm = x @ x.t()
    xy_gm = x @ y.t()
    yy_gm = y @ y.t()

    x_sqnorms = torch.diagonal(xx_gm)
    y_sqnorms = torch.diagonal(yy_gm)

    def pad_first(x: Tensor) -> Tensor:
        return torch.unsqueeze(x, 0)

    def pad_second(x: Tensor) -> Tensor:
        return torch.unsqueeze(x, 1)

    xx_sqnorm = torch.max(-2 * xx_gm + pad_second(x_sqnorms) + pad_first(x_sqnorms), dim=0)[0]
    xy_sqnorm = torch.max(-2 * xy_gm + pad_second(x_sqnorms) + pad_first(y_sqnorms), dim=0)[0]
    yy_sqnorm = torch.max(-2 * yy_gm + pad_second(y_sqnorms) + pad_first(y_sqnorms), dim=0)[0]

    k_xx, k_xy, k_yy = (
        x.new_zeros(xx_sqnorm.shape),
        x.new_zeros(xy_sqnorm.shape),
        x.new_zeros(yy_sqnorm.shape),
    )

    for alpha, wt in zip(scales, wts):
        log_xx = torch.log(1.0 + xx_sqnorm / (2.0 * alpha))
        k_xx += wt * torch.exp(-alpha * log_xx)
        log_xy = torch.log(1.0 + xy_sqnorm / (2.0 * alpha))
        k_xy += wt * torch.exp(-alpha * log_xy)
        log_yy = torch.log(1.0 + yy_sqnorm / (2.0 * alpha))
        k_yy += wt * torch.exp(-alpha * log_yy)

    if add_dot > 0:
        k_xy += add_dot * xy_gm
        k_xx += add_dot * xx_gm
        k_yy += add_dot * yy_gm

    return KernelOut(xx=k_xx, xy=k_xy, yy=k_yy, const_diag=sum(wts))


def _mix_rbf_kernel(
    x: Tensor,
    y: Tensor,
    scales: Sequence[float] | None = None,
    wts: Sequence[float] | None = None,
    add_dot: float = 0.0,
) -> KernelOut:
    """RBF Kernel."""
    scales = (2.0, 5.0, 10.0, 20.0, 40.0, 80.0) if scales is None else scales
    wts = [1.0] * len(scales) if wts is None else wts

    xx_gm = x @ x.t()
    xy_gm = x @ y.t()
    yy_gm = y @ y.t()

    x_sqnorms = torch.diagonal(xx_gm)
    y_sqnorms = torch.diagonal(yy_gm)

    def pad_first(x: Tensor) -> Tensor:
        return torch.unsqueeze(x, 0)

    def pad_second(x: Tensor) -> Tensor:
        return torch.unsqueeze(x, 1)

    xx_sqnorm = -2 * xx_gm + pad_second(x_sqnorms) + pad_first(x_sqnorms)
    xy_sqnorm = -2 * xy_gm + pad_second(x_sqnorms) + pad_first(y_sqnorms)
    yy_sqnorm = -2 * yy_gm + pad_second(y_sqnorms) + pad_first(y_sqnorms)

    k_xx, k_xy, k_yy = (
        x.new_zeros(xx_sqnorm.shape),
        x.new_zeros(xy_sqnorm.shape),
        x.new_zeros(yy_sqnorm.shape),
    )

    for sigma, wt in zip(scales, wts):
        gamma = 1.0 / (2 * sigma ** 2)
        k_xx = wt * torch.exp(-gamma * xx_sqnorm)
        k_xy += wt * torch.exp(-gamma * xy_sqnorm)
        k_yy += wt * torch.exp(-gamma * yy_sqnorm)

    return KernelOut(xx=k_xx, xy=k_xy, yy=k_yy, const_diag=sum(wts))


def _mmd2(kernel: KernelOut, biased: bool = False) -> Tensor:
    m = kernel.xx.size(0)
    n = kernel.yy.size(0)

    if biased:
        return kernel.xx.sum() / (m * m) + kernel.yy.sum() / (n * n) - 2 * kernel.xy.sum() / (m * n)
    if kernel.const_diag != 0.0:
        trace_x = torch.tensor(m)
        trace_y = torch.tensor(n)
    else:
        trace_x = kernel.xx.trace()
        trace_y = kernel.yy.trace()
    return (
        (kernel.xx.sum() - trace_x) / (m * (m - 1))
        + (kernel.yy.sum() - trace_y) / (n * (n - 1))
        - (2 * kernel.xy.sum() / (m * n))
    )


def mmd2(
    x: Tensor,
    y: Tensor,
    kernel: KernelType = KernelType.rbf,
    biased: bool = False,
    scales: Sequence[float] | None = None,
    wts: Sequence[float] | None = None,
    add_dot: float = 0.0,
) -> Tensor:
    """MMD."""
    if x.shape[0] < 2 or y.shape[0] < 2:
        log.warning(
            "Not enough samples in one group to perform MMD. "
            "Returning 0 to not crash, but you should increase the batch size."
        )
        return torch.tensor(0.0)
    if kernel.value == "linear":
        kernel_out = _dot_kernel(x=x, y=y)
    elif kernel.value == "rbf":
        kernel_out = _mix_rbf_kernel(x=x, y=y, scales=scales, wts=wts, add_dot=add_dot)
    elif kernel.value == "rq":
        kernel_out = _mix_rq_kernel(x=x, y=y, scales=scales, wts=wts, add_dot=add_dot)
    else:
        raise NotImplementedError("Only RBF, Linear and RQ kernels implemented.")
    return _mmd2(kernel_out, biased)
