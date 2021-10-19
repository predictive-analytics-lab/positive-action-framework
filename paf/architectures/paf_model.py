"""AIES Model."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict

from conduit.data import TernarySample
import pandas as pd
import pytorch_lightning as pl
from ranzen import implements
import torch
from torch import Tensor, nn
from torch.optim.lr_scheduler import ExponentialLR

__all__ = ["PafModel", "TestStepOut"]

from paf.base_templates import Batch, CfBatch

from . import PafResults
from .model import CycleGan
from .model.model_components import AE, Clf, augment_recons, index_by_s


@dataclass
class TestStepOut:
    enc_z: Tensor
    enc_s_pred: Tensor
    x: Tensor
    s: Tensor
    y: Tensor
    recon: Tensor
    cf_recon: Tensor
    recons_0: Tensor
    recons_1: Tensor
    preds: Tensor
    cycle_loss: Tensor
    preds_0_0: Tensor
    preds_0_1: Tensor
    preds_1_0: Tensor
    preds_1_1: Tensor


class PafModel(pl.LightningModule):
    """Model."""

    def __init__(self, *, encoder: AE | CycleGan, classifier: Clf):
        super().__init__()
        self.enc = encoder
        self.clf = classifier

    @property
    def name(self) -> str:
        return f"PAF_{self.enc.name}"

    @implements(nn.Module)
    def forward(self, *, x: Tensor, s: Tensor) -> dict[str, tuple[Tensor, ...]]:
        recons: list[Tensor] | None = None
        if isinstance(self.enc, AE):
            enc_fwd = self.enc.forward(x=x, s=s)
            recons = enc_fwd.x
        elif isinstance(self.enc, CycleGan):
            cyc_fwd = self.enc.forward(real_a=x, real_b=x)
            recons = [cyc_fwd.fake_a, cyc_fwd.fake_b]
        assert recons is not None
        return self.clf.from_recons(recons)

    @implements(pl.LightningModule)
    def training_step(self, batch: tuple[Tensor, ...], batch_idx: int) -> Tensor:
        """Empty as we do not train the model end to end."""

    @implements(pl.LightningModule)
    def configure_optimizers(self) -> tuple[list[torch.optim.Optimizer], list[ExponentialLR]]:
        """Empty as we do not train the model end to end."""

    @implements(pl.LightningModule)
    def predict_step(self, batch: Batch | CfBatch | TernarySample, *_: Any) -> TestStepOut:
        if isinstance(self.enc, AE):
            enc_fwd = self.enc.forward(batch.x, batch.s)
            enc_z = enc_fwd.z
            enc_s_pred = enc_fwd.s
            recons = enc_fwd.x
        elif isinstance(self.enc, CycleGan):
            cyc_fwd = self.enc.forward(real_a=batch.x, real_b=batch.x)
            recons = [cyc_fwd.fake_a, cyc_fwd.fake_b]
            enc_z = torch.ones_like(batch.x)
            enc_s_pred = torch.ones_like(batch.s)
        else:
            raise NotImplementedError()
        augmented_recons = augment_recons(
            batch.x, self.enc.invert(index_by_s(recons, 1 - batch.s), batch.x), batch.s
        )

        mse_loss_fn = nn.MSELoss(reduction="mean")
        _recons = recons.copy()
        _cyc_loss = torch.tensor(0.0)
        for i in range(1):
            _cfx = self.enc.invert(index_by_s(_recons, 1 - batch.s), batch.x)
            if isinstance(self.enc, AE):
                cf_fwd = self.enc.forward(_cfx, 1 - batch.s)
            else:
                cf_fwd = self.enc.forward(_cfx, _cfx)  # type: ignore[assignment]
            _og = self.enc.invert(index_by_s(cf_fwd.x, batch.s), batch.x)
            cycle_loss = mse_loss_fn(_og, batch.x)
            if i == 0:
                _cyc_loss = cycle_loss
            # self.log(f"Cycle_loss_{i}", cycle_loss)
            if isinstance(self.enc, AE):
                _fwd = self.enc.forward(_og, 1 - batch.s)
            else:
                _fwd = self.enc.forward(_og, _og)  # type: ignore[assignment]
            _recons = _fwd.x

        vals: Dict[str, Tensor] = {
            "enc_z": enc_z,
            "enc_s_pred": enc_s_pred,
            "x": batch.x,
            "s": batch.s,
            "y": batch.y,
            "recon": index_by_s(augmented_recons, batch.s),
            "cf_recon": index_by_s(augmented_recons, 1 - batch.s),
            "recons_0": self.enc.invert(recons[0], batch.x),
            "recons_1": self.enc.invert(recons[1], batch.x),
            "preds": self.clf.threshold(
                index_by_s(self.clf.forward(x=batch.x, s=batch.s)[-1], batch.s)
            ),
            "cycle_loss": _cyc_loss,
        }

        for i, recon in enumerate(augmented_recons):
            clf_out = self.clf.forward(x=recon, s=torch.ones_like(batch.s) * i)
            vals.update({f"preds_{i}_{j}": self.clf.threshold(clf_out.y[j]) for j in range(2)})
        return TestStepOut(**vals)

    @staticmethod
    def collate_results(outputs: list[TestStepOut]) -> PafResults:
        preds_0_0 = torch.cat([_r.preds_0_0 for _r in outputs], 0)
        preds_0_1 = torch.cat([_r.preds_0_1 for _r in outputs], 0)
        preds_1_0 = torch.cat([_r.preds_1_0 for _r in outputs], 0)
        preds_1_1 = torch.cat([_r.preds_1_1 for _r in outputs], 0)
        s = torch.cat([_r.s for _r in outputs], 0)
        preds = torch.cat([_r.preds for _r in outputs], 0)

        return PafResults(
            enc_z=torch.cat([_r.enc_z for _r in outputs], 0),
            enc_s_pred=torch.cat([_r.enc_s_pred for _r in outputs], 0),
            s=s,
            x=torch.cat([_r.x for _r in outputs], 0),
            y=torch.cat([_r.y for _r in outputs], 0),
            recon=torch.cat([_r.recon for _r in outputs], 0),
            cf_x=torch.cat([_r.cf_recon for _r in outputs], 0),
            recons_0=torch.cat([_r.recons_0 for _r in outputs], 0),
            recons_1=torch.cat([_r.recons_1 for _r in outputs], 0),
            preds=preds,
            preds_0_0=preds_0_0,
            preds_0_1=preds_0_1,
            preds_1_0=preds_1_0,
            preds_1_1=preds_1_1,
            pd_results=pd.DataFrame(
                torch.cat(
                    [
                        preds_0_0,
                        preds_0_1,
                        preds_1_0,
                        preds_1_1,
                        s.unsqueeze(-1),
                        preds,
                    ],
                    dim=1,
                )
                .to(torch.long)
                .cpu()
                .numpy(),
                columns=["s1_0_s2_0", "s1_0_s2_1", "s1_1_s2_0", "s1_1_s2_1", "true_s", "actual"],
            ),
            cycle_loss=torch.tensor(sum(_r.cycle_loss for _r in outputs) / s.shape[0]),
        )
