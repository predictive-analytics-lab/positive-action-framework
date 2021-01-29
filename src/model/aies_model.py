"""AIES Model."""
from typing import Dict, List, Tuple

import pandas as pd
import torch
from ethicml import implements
from pytorch_lightning import LightningModule
from torch import Tensor, nn
from torch.optim.lr_scheduler import ExponentialLR

from src.model.aies_properties import AiesProperties
from src.model.classifier_model import Clf
from src.model.encoder_model import AE
from src.model.model_utils import index_by_s


class AiesModel(AiesProperties):
    """Model."""

    def __init__(self, encoder: AE, classifier: Clf):
        super().__init__()
        self.enc = encoder
        self.clf = classifier

    @implements(nn.Module)
    def forward(self, x: Tensor, s: Tensor) -> Dict[str, Tuple[Tensor, ...]]:
        enc_z, enc_s_pred, recons = self.enc(x, s)
        return self.clf.from_recons(recons)

    @implements(LightningModule)
    def training_step(self, batch: Tuple[Tensor, ...], batch_idx: int) -> Tensor:
        """This is empty as we do not train the model end to end."""

    @implements(LightningModule)
    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[ExponentialLR]]:
        """This is empty as we do not train the model end to end."""

    @implements(LightningModule)
    def test_step(self, batch: Tuple[Tensor, ...], batch_idx: int) -> Dict[str, Tensor]:
        if self.enc.cf_model:
            x, s, y, cf_x, cf_s, cf_y, _ = batch
        else:
            x, s, y, _ = batch

        enc_z, enc_s_pred, recons = self.enc(x, s)

        self.enc.invert(index_by_s(recons, 1 - s))

        augmented_recons = recons  # augment_recons(x, cf_recons, s)

        to_return = {
            "enc_z": enc_z,
            "enc_s_pred": enc_s_pred,
            "x": x,
            "s": s,
            "y": y,
            "recon": index_by_s(augmented_recons, s),
            "recons_0": self.enc.invert(recons[0]),
            "recons_1": self.enc.invert(recons[1]),
            "preds": self.clf.threshold(index_by_s(self.clf(x, s)[-1], s)),
        }

        for i, recon in enumerate(augmented_recons):
            clf_z, clf_s_pred, preds = self.clf(recon, torch.ones_like(s) * i)
            # to_return[f"preds_{i}"] = self.clf.threshold(index_by_s(preds, s))
            to_return[f"preds_{i}_0"] = self.clf.threshold(preds[0])
            to_return[f"preds_{i}_1"] = self.clf.threshold(preds[1])

        # if self.enc.cf_model:
        #     to_return["true_s1_0_s2_0"] = self.clf.threshold(index_by_s(preds, cf_s))

        return to_return

    @implements(LightningModule)
    def test_epoch_end(self, output_results: List[Dict[str, Tensor]]) -> None:
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
