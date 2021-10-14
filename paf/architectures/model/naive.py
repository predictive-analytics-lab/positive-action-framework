from __future__ import annotations
from typing import Any

from conduit.data import TernarySample
import pytorch_lightning as pl
from torch import Tensor, nn
import torch.optim
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from paf.base_templates.dataset_utils import Batch, CfBatch

__all__ = ["NaiveModel"]


class NaiveModel(pl.LightningModule):
    def __init__(self, in_size: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_size, 50), nn.Linear(50, 50), nn.Linear(50, 1))
        self._loss = nn.BCEWithLogitsLoss()
        self.learning_rate = 1e-2

    def training_step(self, batch: Batch | CfBatch | TernarySample, *_: Any) -> Tensor:
        logits = self.net(batch.x)
        return self._loss(logits, batch.y.view(-1, 1))

    def test_step(self, batch: Batch | CfBatch | TernarySample, *_: Any) -> dict[str, Tensor]:
        return {"preds": self.forward(batch.x)}

    def test_epoch_end(self, outputs: dict[str, Tensor]) -> None:
        """Not used anywhere, but needed for super."""

    def configure_optimizers(
        self,
    ) -> tuple[list[torch.optim.Optimizer], list[CosineAnnealingWarmRestarts]]:
        optim = AdamW(self.net.parameters(), lr=self.learning_rate)
        sched = CosineAnnealingWarmRestarts(optimizer=optim, T_0=1, T_mult=2)
        return [optim], [sched]

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
