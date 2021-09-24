"""AIES Model."""
from __future__ import annotations
from typing import Any, NamedTuple

from kit import implements
import pandas as pd
import pytorch_lightning as pl
import torch
from torch import Tensor, nn
from torch.optim.lr_scheduler import ExponentialLR

__all__ = ["PafModel", "TestStepOut"]

from paf.base_templates import Batch, CfBatch

from .model import CycleGan
from .model.model_components import AE, Clf, augment_recons, index_by_s


class PafModel(pl.LightningModule):
    """Model."""

    recon_0: Tensor
    recon_1: Tensor
    pd_results: pd.DataFrame
    all_enc_z: Tensor
    all_enc_s_pred: Tensor
    all_s: Tensor
    all_x: Tensor
    all_y: Tensor
    all_recon: Tensor
    all_preds: Tensor

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
            enc_fwd = self.enc.forward(x, s)
            recons = enc_fwd.x
        elif isinstance(self.enc, CycleGan):
            cyc_fwd = self.enc.forward(x, x)
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
    def test_step(self, batch: Batch | CfBatch, *_: Any) -> TestStepOut:
        if isinstance(self.enc, AE):
            enc_fwd = self.enc.forward(batch.x, batch.s)
            enc_z = enc_fwd.z
            enc_s_pred = enc_fwd.s
            recons = enc_fwd.x
        elif isinstance(self.enc, CycleGan):
            cyc_fwd = self.enc.forward(batch.x, batch.x)
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
        _cyc_loss = 0.0
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

        vals = {
            "enc_z": enc_z,
            "enc_s_pred": enc_s_pred,
            "x": batch.x,
            "s": batch.s,
            "y": batch.y,
            "recon": index_by_s(augmented_recons, batch.s),
            "recons_0": self.enc.invert(recons[0], batch.x),
            "recons_1": self.enc.invert(recons[1], batch.x),
            "preds": self.clf.threshold(
                index_by_s(self.clf.forward(batch.x, batch.s)[-1], batch.s)
            ),
            "cycle_loss": _cyc_loss,
        }

        for i, recon in enumerate(augmented_recons):
            clf_out = self.clf.forward(recon, torch.ones_like(batch.s) * i)
            vals.update({f"preds_{i}_{j}": self.clf.threshold(clf_out.y[j]) for j in range(2)})
        return TestStepOut(**vals)

    @implements(pl.LightningModule)
    def test_epoch_end(self, outputs: list[TestStepOut]) -> None:
        self.all_enc_z = torch.cat([_r.enc_z for _r in outputs], 0)
        self.all_enc_s_pred = torch.cat([_r.enc_s_pred for _r in outputs], 0)
        self.all_s = torch.cat([_r.s for _r in outputs], 0)
        self.all_x = torch.cat([_r.x for _r in outputs], 0)
        self.all_y = torch.cat([_r.y for _r in outputs], 0)
        self.all_recon = torch.cat([_r.recon for _r in outputs], 0)
        self.recon_0 = torch.cat([_r.recons_0 for _r in outputs], 0)
        self.recon_1 = torch.cat([_r.recons_1 for _r in outputs], 0)
        self.all_preds = torch.cat([_r.preds for _r in outputs], 0)

        all_s0_s0_preds = torch.cat([_r.preds_0_0 for _r in outputs], 0)
        all_s0_s1_preds = torch.cat([_r.preds_0_1 for _r in outputs], 0)
        all_s1_s0_preds = torch.cat([_r.preds_1_0 for _r in outputs], 0)
        all_s1_s1_preds = torch.cat([_r.preds_1_1 for _r in outputs], 0)

        self.pd_results = pd.DataFrame(
            torch.cat(
                [
                    all_s0_s0_preds,
                    all_s0_s1_preds,
                    all_s1_s0_preds,
                    all_s1_s1_preds,
                    self.all_s.unsqueeze(-1),
                    self.all_preds,
                ],
                dim=1,
            )
            .to(torch.long)
            .cpu()
            .numpy(),
            columns=["s1_0_s2_0", "s1_0_s2_1", "s1_1_s2_0", "s1_1_s2_1", "true_s", "actual"],
        )
        cycle_loss: Tensor = sum(_r.cycle_loss for _r in outputs) / self.all_y.shape[0]  # type: ignore[assignment]
        self.log(name="cycle_loss", value=cycle_loss.item(), logger=True)


class TestStepOut(NamedTuple):
    enc_z: Tensor
    enc_s_pred: Tensor
    x: Tensor
    s: Tensor
    y: Tensor
    recon: Tensor
    recons_0: Tensor
    recons_1: Tensor
    preds: Tensor
    cycle_loss: Tensor
    preds_0_0: Tensor
    preds_0_1: Tensor
    preds_1_0: Tensor
    preds_1_1: Tensor
