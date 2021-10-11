from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from conduit.fair.data import EthicMlDataModule
import pandas as pd
import pytorch_lightning as pl
import pytorch_lightning.utilities.types as plut
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from paf.architectures import NnResults
from paf.architectures.model.model_components import augment_recons
from paf.base_templates import BaseDataModule
from paf.base_templates.dataset_utils import Batch, CfBatch

__all__ = ["NearestNeighbourModel", "NnStepOut"]


@dataclass
class NnStepOut:
    cf_preds: Tensor
    cf_x: Tensor
    preds: Tensor
    x: Tensor
    s: Tensor
    y: Tensor
    recons_0: Tensor
    recons_1: Tensor


class NearestNeighbourModel(pl.LightningModule):
    name = "NearestNeighbour"
    all_preds: Tensor
    all_cf_preds: Tensor
    all_cf_x: Tensor
    all_x: Tensor
    all_s: Tensor
    all_y: Tensor
    pd_results: pd.DataFrame

    def __init__(self, clf_model: nn.Module, data: BaseDataModule | EthicMlDataModule):
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

    def training_step(self, *_: Any) -> plut.STEP_OUTPUT:
        ...

    def predict_step(self, batch: Batch | CfBatch, batch_idx: int, *_: Any) -> NnStepOut | None:
        cf_feats, cf_outcome = self.forward(test_features=batch.x, sens_label=batch.s)
        preds = (self.clf.forward(batch.x) >= 0).long()

        augmented_recons = augment_recons(batch.x, cf_feats, batch.s)

        return NnStepOut(
            cf_preds=cf_outcome,
            cf_x=cf_feats,
            preds=preds,
            x=batch.x,
            s=batch.s,
            y=batch.y,
            recons_0=augmented_recons[0],
            recons_1=augmented_recons[1],
        )

    @staticmethod
    def collate_results(outputs: list[NnStepOut]) -> NnResults:
        all_preds = torch.cat([_r.preds for _r in outputs], 0)
        all_cf_preds = torch.cat([_r.cf_preds for _r in outputs], 0)
        all_cf_x = torch.cat([_r.cf_x for _r in outputs], 0)

        all_s = torch.cat([_r.s for _r in outputs], 0)
        all_x = torch.cat([_r.x for _r in outputs], 0)
        all_y = torch.cat([_r.y for _r in outputs], 0)
        stacked = []
        for _s, pred, cfpred in zip(all_s, all_preds, all_cf_preds):
            if _s == 0:
                stacked.append((pred, cfpred))
            else:
                stacked.append((cfpred, pred))

        stacked = torch.tensor(stacked)

        return NnResults(
            cf_preds=all_cf_preds,
            cf_x=all_cf_x,
            preds=all_preds,
            x=all_x,
            s=all_s,
            y=all_y,
            pd_results=pd.DataFrame(
                torch.cat(
                    [
                        stacked[:, 0].unsqueeze(-1).long(),
                        stacked[:, 1].unsqueeze(-1).long(),
                        all_s.unsqueeze(-1).long(),
                        all_preds,
                    ],
                    dim=1,
                )
                .cpu()
                .numpy(),
                columns=["s1_0_s2_0", "s1_1_s2_1", "true_s", "actual"],
            ),
            recons_0=torch.cat([_r.recons_0 for _r in outputs], 0),
            recons_1=torch.cat([_r.recons_1 for _r in outputs], 0),
        )

    def configure_optimizers(
        self,
    ) -> tuple[list[torch.optim.Optimizer], list[CosineAnnealingWarmRestarts]]:
        ...
