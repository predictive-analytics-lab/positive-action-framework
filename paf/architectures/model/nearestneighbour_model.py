from __future__ import annotations
from typing import NamedTuple

from conduit.types import Stage
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from paf.base_templates.dataset_utils import Batch, CfBatch
from paf.mmd import KernelType, mmd2

__all__ = ["NearestNeighbourModel", "NnStepOut"]


class NearestNeighbourModel(pl.LightningModule):
    name = "NearestNeighbour"
    all_preds: Tensor
    all_cf_preds: Tensor
    all_cf_x: Tensor
    all_x: Tensor
    all_s: Tensor
    all_y: Tensor
    pd_results: pd.DataFrame

    def __init__(self, clf_model: nn.Module, data: pl.LightningDataModule):
        super().__init__()
        self.clf = clf_model
        self.train_features = torch.tensor(data.train_datatuple.x.values)
        self.train_sens = torch.tensor(data.train_datatuple.s.values)
        self.train_labels = torch.tensor(data.train_datatuple.y.values)

        self.train_features = nn.Parameter(
            F.normalize(self.train_features.detach(), dim=1, p=2), requires_grad=False
        ).float()
        self.train_labels = nn.Parameter(self.train_labels.detach(), requires_grad=False).float()

    def forward(self, *, test_features: Tensor, sens_label: Tensor) -> tuple[Tensor, Tensor]:
        test_features = F.normalize(test_features, dim=1, p=2)

        features = []
        labels = []

        for point, s_label in zip(test_features, sens_label):
            sim = point @ self.train_features[(self.train_sens != s_label).squeeze(-1)].t()
            features.append(
                self.train_features[(self.train_sens != s_label).squeeze(-1)][sim.argmax(-1)]
            )
            labels.append(
                self.train_labels[(self.train_sens != s_label).squeeze(-1)][sim.argmax(-1)]
            )

        return torch.stack(features, dim=0), torch.stack(labels, dim=0)

    def training_step(self, batch: Batch | CfBatch, batch_idx: int) -> STEP_OUTPUT:
        ...

    def test_step(self, batch: Batch | CfBatch, batch_idx: int) -> NnStepOut | None:
        cf_feats, cf_outcome = self.forward(test_features=batch.x, sens_label=batch.s)
        preds = (self.clf.forward(batch.x) >= 0).long()

        return NnStepOut(
            cf_preds=cf_outcome,
            cf_x=cf_feats,
            preds=preds,
            x=batch.x,
            s=batch.s,
            y=batch.y,
        )

    def on_test_epoch_end(self) -> None:
        kernel = KernelType.LINEAR

        recon_mmd = mmd2(self.all_x, self.all_cf_x, kernel=kernel)
        s0_dist_mmd = mmd2(
            self.all_x[self.all_s == 0],
            self.all_cf_x[self.all_s == 1],
            kernel=kernel,
        )
        s1_dist_mmd = mmd2(
            self.all_x[self.all_s == 1],
            self.all_cf_x[self.all_s == 0],
            kernel=kernel,
        )

        self.log(
            name=f"Logging/{Stage.test}/MMD",
            value=round(recon_mmd.item(), 5),
            logger=True,
        )

        self.log(
            name=f"Logging/{Stage.test}/MMD S0 vs Cf",
            value=round(s0_dist_mmd.item(), 5),
            logger=True,
        )

        self.log(
            name=f"Logging/{Stage.test}/MMD S1 vs Cf",
            value=round(s1_dist_mmd.item(), 5),
            logger=True,
        )

    def test_epoch_end(self, outputs: list[NnStepOut]) -> None:
        self.all_preds = torch.cat([_r.preds for _r in outputs], 0)
        self.all_cf_preds = torch.cat([_r.cf_preds for _r in outputs], 0)
        self.all_cf_x = torch.cat([_r.cf_x for _r in outputs], 0)

        self.all_s = torch.cat([_r.s for _r in outputs], 0)
        self.all_x = torch.cat([_r.x for _r in outputs], 0)
        self.all_y = torch.cat([_r.y for _r in outputs], 0)
        stacked = []
        for _s, pred, cfpred in zip(self.all_s, self.all_preds, self.all_cf_preds):
            if _s == 0:
                stacked.append((pred, cfpred))
            else:
                stacked.append((cfpred, pred))

        stacked = torch.tensor(stacked)

        self.pd_results = pd.DataFrame(
            torch.cat(
                [
                    stacked[:, 0].unsqueeze(-1).long(),
                    stacked[:, 1].unsqueeze(-1).long(),
                    self.all_s.unsqueeze(-1).long(),
                    self.all_preds,
                ],
                dim=1,
            )
            .cpu()
            .numpy(),
            columns=["s1_0_s2_0", "s1_1_s2_1", "true_s", "actual"],
        )

    def configure_optimizers(
        self,
    ) -> tuple[list[torch.optim.Optimizer], list[CosineAnnealingWarmRestarts]]:
        ...


class NnStepOut(NamedTuple):
    cf_preds: Tensor
    cf_x: Tensor
    preds: Tensor
    x: Tensor
    s: Tensor
    y: Tensor
