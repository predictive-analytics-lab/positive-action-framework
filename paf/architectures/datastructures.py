from dataclasses import dataclass

import pandas as pd
from torch import Tensor

__all__ = ["Results", "PafResults", "NnResults"]


@dataclass
class Results:
    pd_results: pd.DataFrame
    x: Tensor
    s: Tensor
    y: Tensor
    cf_x: Tensor
    preds: Tensor


@dataclass
class PafResults(Results):
    enc_z: Tensor
    enc_s_pred: Tensor
    recons_0: Tensor
    recons_1: Tensor
    preds_0_0: Tensor
    preds_0_1: Tensor
    preds_1_0: Tensor
    preds_1_1: Tensor
    recon: Tensor
    cycle_loss: Tensor


@dataclass
class NnResults(Results):
    cf_preds: Tensor
