from typing import Dict, Tuple

import pytorch_lightning as pl
from torch import Tensor, nn


class NaiveModel(pl.LightningModule):
    def __init__(self, in_size: int, num_pos_action: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_size, 1))
        self._loss = nn.BCEWithLogitsLoss()
        self.num_pos_action = num_pos_action

    def training_step(self, batch: Tuple[Tensor, ...], batch_idx: int) -> Tensor:
        x, s, y = batch
        logits = self.net(x)
        return self._loss(logits, y)

    def test_step(self, batch: Tuple[Tensor, ...], batch_dx: int) -> Dict[str, Tensor]:
        x, s, y = batch
        return {"preds": self(x, s)}

    def test_epoch_end(self, outputs: Dict[str, Tensor]) -> None:
        """Not used anywhere, but needed for super."""

    def forward(self, x: Tensor, s: Tensor) -> Tensor:
        return self.net(x)
