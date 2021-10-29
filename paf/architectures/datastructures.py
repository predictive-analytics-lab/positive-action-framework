from dataclasses import dataclass
from typing import NamedTuple

import pandas as pd
from torch import Tensor

__all__ = ["Results", "PafResults", "NnResults", "MmdReportingResults"]


@dataclass
class Results:
    pd_results: pd.DataFrame
    x: Tensor
    s: Tensor
    y: Tensor
    cf_x: Tensor
    preds: Tensor
    recons_0: Tensor
    recons_1: Tensor


@dataclass
class PafResults(Results):
    enc_z: Tensor
    clf_z0: Tensor
    clf_z1: Tensor
    clf_z: Tensor
    enc_s_pred: Tensor
    preds_0_0: Tensor
    preds_0_1: Tensor
    preds_1_0: Tensor
    preds_1_1: Tensor
    recon: Tensor
    cycle_loss: Tensor
    cyc_vals: pd.DataFrame


@dataclass
class NnResults(Results):
    cf_preds: Tensor


class MmdReportingResults(NamedTuple):
    recon: Tensor
    cf_recon: Tensor
    s0_dist: Tensor
    s1_dist: Tensor
