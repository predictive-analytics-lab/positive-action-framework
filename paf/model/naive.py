from typing import Dict, Union

import pytorch_lightning as pl
from torch import Tensor, nn
from torch.optim import AdamW

from paf.base_templates.dataset_utils import Batch, CfBatch


class NaiveModel(pl.LightningModule):
    def __init__(self, in_size: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_size, 1))
        self._loss = nn.BCEWithLogitsLoss()
        self.learning_rate = 1e-3

    def training_step(self, batch: Union[Batch, CfBatch], batch_idx: int) -> Tensor:
        logits = self.net(batch.x)
        return self._loss(logits, batch.y.view(-1, 1))

    def test_step(self, batch: Union[Batch, CfBatch], batch_dx: int) -> Dict[str, Tensor]:
        return {"preds": self(batch.x)}

    def test_epoch_end(self, outputs: Dict[str, Tensor]) -> None:
        """Not used anywhere, but needed for super."""

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.learning_rate)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
