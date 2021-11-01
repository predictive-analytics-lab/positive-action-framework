from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from conduit.data import TernarySample
from conduit.fair.data import EthicMlDataModule
import numpy as np
import pandas as pd
import pytorch_lightning.utilities.types as plut
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from paf.architectures.model.model_components import (
    CommonModel,
    augment_recons,
    index_by_s,
)
from paf.base_templates import BaseDataModule
from paf.base_templates.dataset_utils import Batch, CfBatch

__all__ = ["NearestNeighbour", "NnStepOut"]


@dataclass
class NnStepOut:
    cf_x: Tensor
    x: Tensor
    s: Tensor
    recons_0: Tensor
    recons_1: Tensor


@dataclass
class NnFwd:
    x: list[Tensor]


class NearestNeighbour(CommonModel):
    name = "NearestNeighbour"
    all_preds: Tensor
    all_cf_preds: Tensor
    all_cf_x: Tensor
    all_x: Tensor
    all_s: Tensor
    all_y: Tensor
    pd_results: pd.DataFrame

    def __init__(self) -> None:
        super().__init__(name="NN")

    def build(
        self,
        *,
        num_s: int,
        data_dim: int,
        s_dim: int,
        cf_available: bool,
        feature_groups: dict[str, list[slice]],
        outcome_cols: list[str],
        data: BaseDataModule | EthicMlDataModule,
        indices: list[str] | None,
    ) -> None:
        _ = (num_s, data_dim, s_dim, cf_available, feature_groups, outcome_cols, indices)
        self.train_features = torch.tensor(data.train_datatuple.x.values)
        self.train_sens = torch.tensor(data.train_datatuple.s.values)
        self.train_labels = torch.tensor(data.train_datatuple.y.values)

        self.train_features = nn.Parameter(
            F.normalize(self.train_features.detach(), dim=1, p=2), requires_grad=False
        ).float()

    def forward(self, *, x: Tensor, s: Tensor) -> NnFwd:
        x = F.normalize(x, dim=1, p=2)

        features = []

        for point, s_label in zip(x, s):
            sim = point @ self.train_features[(self.train_sens != s_label).squeeze(-1)].t().to(
                point.device
            )
            features.append(
                self.train_features[(self.train_sens != s_label).squeeze(-1)][sim.argmax(-1)]
            )
        return NnFwd(x=augment_recons(x, torch.stack(features, dim=0), s))

    def training_step(self, *_: Any) -> plut.STEP_OUTPUT:
        ...

    def predict_step(
        self, batch: Batch | CfBatch | TernarySample, batch_idx: int, *_: Any
    ) -> NnStepOut | None:
        recon_list = self.forward(x=batch.x, s=batch.s)

        return NnStepOut(
            cf_x=index_by_s(recon_list.x, 1 - batch.s),
            x=index_by_s(recon_list.x, batch.s),
            s=batch.s,
            recons_0=recon_list.x[0],
            recons_1=recon_list.x[1],
        )

    def configure_optimizers(
        self,
    ) -> tuple[list[torch.optim.Optimizer], list[CosineAnnealingWarmRestarts]]:
        ...

    def get_recon(self, dataloader: DataLoader) -> np.ndarray:
        raise NotImplementedError("This shouldn't be called. Only implementing for the abc.")

    def invert(self, z: Tensor, x: Tensor) -> Tensor:
        return z
