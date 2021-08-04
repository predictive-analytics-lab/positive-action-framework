"""AIES Model."""
from __future__ import annotations

from kit import implements
import pandas as pd
from pytorch_lightning import LightningModule
import torch
from torch import Tensor, nn
from torch.optim.lr_scheduler import ExponentialLR

from paf.base_templates.dataset_utils import Batch, CfBatch
from paf.log_progress import do_log
from paf.model import CycleGan
from paf.model.aies_properties import AiesProperties
from paf.model.classifier_model import Clf
from paf.model.encoder_model import AE
from paf.model.model_utils import augment_recons, index_by_s


class AiesModel(AiesProperties):
    """Model."""

    def __init__(self, encoder: AE, classifier: Clf):
        super().__init__()
        self.enc = encoder
        self.clf = classifier

    @implements(nn.Module)
    def forward(self, x: Tensor, s: Tensor) -> dict[str, tuple[Tensor, ...]]:
        enc_z, enc_s_pred, recons = self.enc(x, s)
        return self.clf.from_recons(recons)

    @implements(LightningModule)
    def training_step(self, batch: tuple[Tensor, ...], batch_idx: int) -> Tensor:
        """This is empty as we do not train the model end to end."""

    @implements(LightningModule)
    def configure_optimizers(self) -> tuple[list[torch.optim.Optimizer], list[ExponentialLR]]:
        """This is empty as we do not train the model end to end."""

    @implements(LightningModule)
    def test_step(self, batch: Batch | CfBatch, batch_idx: int) -> dict[str, Tensor]:
        if isinstance(self.enc, AE):
            enc_z, enc_s_pred, recons = self.enc(batch.x, batch.s)
        elif isinstance(self.enc, CycleGan):
            recons = list(self.enc(batch.x, batch.x))
            enc_z = torch.ones_like(batch.x)
            enc_s_pred = torch.ones_like(batch.s)
        else:
            raise NotImplementedError()
        cf_recons = self.enc.invert(index_by_s(recons, 1 - batch.s), batch.x)
        augmented_recons = augment_recons(batch.x, cf_recons, batch.s)

        cfs = index_by_s(augmented_recons, 1 - batch.s)
        # _, _, _recons = self.enc(batch.x, batch.s)
        _cf_recons = self.enc.invert(index_by_s(recons, 1 - batch.s), cfs)
        _augmented_recons = augment_recons(cfs, _cf_recons, batch.s)

        cycle_loss = nn.MSELoss()(index_by_s(_augmented_recons, batch.s), batch.x)

        to_return = {
            "enc_z": enc_z,
            "enc_s_pred": enc_s_pred,
            "x": batch.x,
            "s": batch.s,
            "y": batch.y,
            "recon": index_by_s(augmented_recons, batch.s),
            "recons_0": self.enc.invert(recons[0], batch.x),
            "recons_1": self.enc.invert(recons[1], batch.x),
            "preds": self.clf.threshold(index_by_s(self.clf(batch.x, batch.s)[-1], batch.s)),
            "cycle_loss": cycle_loss,
        }

        for i, recon in enumerate(augmented_recons):
            clf_z, clf_s_pred, preds = self.clf(recon, torch.ones_like(batch.s) * i)
            to_return[f"preds_{i}_0"] = self.clf.threshold(preds[0])
            to_return[f"preds_{i}_1"] = self.clf.threshold(preds[1])

        return to_return

    @implements(LightningModule)
    def test_epoch_end(self, output_results: list[dict[str, Tensor]]) -> None:
        self.all_enc_z = torch.cat([_r["enc_z"] for _r in output_results], 0)
        self.all_enc_s_pred = torch.cat([_r["enc_s_pred"] for _r in output_results], 0)
        self.all_s = torch.cat([_r["s"] for _r in output_results], 0)
        self.all_x = torch.cat([_r["x"] for _r in output_results], 0)
        self.all_y = torch.cat([_r["y"] for _r in output_results], 0)
        self.all_recon = torch.cat([_r["recon"] for _r in output_results], 0)
        self.recon_0 = torch.cat([_r["recons_0"] for _r in output_results], 0)
        self.recon_1 = torch.cat([_r["recons_1"] for _r in output_results], 0)
        self.all_preds = torch.cat([_r["preds"] for _r in output_results], 0)

        all_s0_s0_preds = torch.cat([_r["preds_0_0"] for _r in output_results], 0)
        all_s0_s1_preds = torch.cat([_r["preds_0_1"] for _r in output_results], 0)
        all_s1_s0_preds = torch.cat([_r["preds_1_0"] for _r in output_results], 0)
        all_s1_s1_preds = torch.cat([_r["preds_1_1"] for _r in output_results], 0)

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

        do_log(
            "cycle_loss",
            (sum(_r["cycle_loss"] for _r in output_results) / self.all_y.shape[0]).item(),
            self.logger,
        )
