"""AIES Model."""
from __future__ import annotations
from typing import NamedTuple

from kit import implements
import pandas as pd
from pytorch_lightning import LightningModule
import torch
from torch import Tensor, nn
from torch.optim.lr_scheduler import ExponentialLR

from paf.base_templates.dataset_utils import Batch, CfBatch
from paf.model import CycleGan
from paf.model.aies_properties import AiesProperties
from paf.model.classifier_model import Clf
from paf.model.encoder_model import AE
from paf.model.model_utils import augment_recons, index_by_s


class AiesModel(AiesProperties):
    """Model."""

    def __init__(self, encoder: AE | CycleGan, classifier: Clf):
        super().__init__()
        self.enc = encoder
        self.clf = classifier

    @implements(nn.Module)
    def forward(self, x: Tensor, s: Tensor) -> dict[str, tuple[Tensor, ...]]:
        if isinstance(self.enc, AE):
            enc_fwd = self.enc.forward(x, s)
            recons = enc_fwd.x
        elif isinstance(self.enc, CycleGan):
            cyc_fwd = self.enc.forward(x, x)
            recons = [cyc_fwd.fake_b, cyc_fwd.fake_a]
        return self.clf.from_recons(recons)

    @implements(LightningModule)
    def training_step(self, batch: tuple[Tensor, ...], batch_idx: int) -> Tensor:
        """This is empty as we do not train the model end to end."""

    @implements(LightningModule)
    def configure_optimizers(self) -> tuple[list[torch.optim.Optimizer], list[ExponentialLR]]:
        """This is empty as we do not train the model end to end."""

    @implements(LightningModule)
    def test_step(self, batch: Batch | CfBatch, batch_idx: int) -> TestStepOut:
        if isinstance(self.enc, AE):
            enc_fwd = self.enc.forward(batch.x, batch.s)
            enc_z = enc_fwd.z
            enc_s_pred = enc_fwd.s
            recons = enc_fwd.x
        elif isinstance(self.enc, CycleGan):
            cyc_fwd = self.enc.forward(batch.x, batch.x)
            recons = [cyc_fwd.fake_b, cyc_fwd.fake_a]
            enc_z = torch.ones_like(batch.x)
            enc_s_pred = torch.ones_like(batch.s)
        else:
            raise NotImplementedError()
        cf_recons = self.enc.invert(index_by_s(recons, 1 - batch.s), batch.x)
        augmented_recons = augment_recons(batch.x, cf_recons, batch.s)

        mse_loss_fn = nn.MSELoss(reduction="mean")
        for i in range(100):
            cfs = index_by_s(augmented_recons, 1 - batch.s)
            _cf_recons = self.enc.invert(index_by_s(recons, 1 - batch.s), cfs)
            _augmented_recons = augment_recons(cfs, _cf_recons, batch.s)
            cycle_loss = mse_loss_fn(index_by_s(_augmented_recons, batch.s), batch.x)
            self.log(f"Cycle_loss_{i}", cycle_loss)

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
            "cycle_loss": cycle_loss,
        }

        for i, recon in enumerate(augmented_recons):
            clf_out = self.clf.forward(recon, torch.ones_like(batch.s) * i)
            vals.update({f"preds_{i}_{j}": self.clf.threshold(clf_out.y[j]) for j in range(2)})
        return TestStepOut(**vals)

    @implements(LightningModule)
    def test_epoch_end(self, output_results: list[TestStepOut]) -> None:
        self.all_enc_z = torch.cat([_r.enc_z for _r in output_results], 0)
        self.all_enc_s_pred = torch.cat([_r.enc_s_pred for _r in output_results], 0)
        self.all_s = torch.cat([_r.s for _r in output_results], 0)
        self.all_x = torch.cat([_r.x for _r in output_results], 0)
        self.all_y = torch.cat([_r.y for _r in output_results], 0)
        self.all_recon = torch.cat([_r.recon for _r in output_results], 0)
        self.recon_0 = torch.cat([_r.recons_0 for _r in output_results], 0)
        self.recon_1 = torch.cat([_r.recons_1 for _r in output_results], 0)
        self.all_preds = torch.cat([_r.preds for _r in output_results], 0)

        all_s0_s0_preds = torch.cat([_r.preds_0_0 for _r in output_results], 0)
        all_s0_s1_preds = torch.cat([_r.preds_0_1 for _r in output_results], 0)
        all_s1_s0_preds = torch.cat([_r.preds_1_0 for _r in output_results], 0)
        all_s1_s1_preds = torch.cat([_r.preds_1_1 for _r in output_results], 0)

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
            .cpu()
            .numpy(),
            columns=["s1_0_s2_0", "s1_0_s2_1", "s1_1_s2_0", "s1_1_s2_1", "true_s", "actual"],
        )
        cycle_loss: Tensor = sum(_r.cycle_loss for _r in output_results) / self.all_y.shape[0]  # type: ignore[assignment]
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
