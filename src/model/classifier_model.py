"""Encoder model."""
from typing import Dict, List, Tuple

import numpy as np
import torch
from ethicml import implements
from pytorch_lightning import LightningModule
from torch import Tensor, cat, nn, no_grad
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from src.config_classes.dataclasses import ModelConfig
from src.mmd import mmd2
from src.model.blocks import block, mid_blocks
from src.model.common_model import CommonModel
from src.model.model_utils import grad_reverse, index_by_s
from src.utils import do_log, make_plot


class BaseModel(nn.Module):
    """Base AE Model."""

    def __init__(self, *, in_size: int, hid_size: int, out_size: int, blocks: int):
        super().__init__()
        if blocks == 0:
            self.hid = nn.Identity()
            self.out = nn.Linear(in_size, out_size)
        else:
            _blocks = [block(in_dim=in_size, out_dim=hid_size)] + mid_blocks(latent_dim=hid_size, blocks=blocks)
            self.hid = nn.Sequential(*_blocks)
            self.out = nn.Linear(hid_size, out_size)
        nn.init.xavier_normal_(self.out.weight)

    @implements(nn.Module)
    def forward(self, x: Tensor) -> Tensor:
        z = self.hid(x)
        return self.out(z)


class Encoder(BaseModel):
    """AE Shared Encoder."""

    def __init__(self, *, in_size: int, latent_dim: int, blocks: int, hid_multiplier: int):
        super().__init__(in_size=in_size, hid_size=latent_dim * hid_multiplier, out_size=latent_dim, blocks=blocks)


class Adversary(BaseModel):
    """AE Adversary head."""

    def __init__(self, *, latent_dim: int, out_size: int, blocks: int, hid_multiplier: int):
        super().__init__(in_size=latent_dim, hid_size=latent_dim * hid_multiplier, out_size=out_size, blocks=blocks)

    @implements(nn.Module)
    def forward(self, z: Tensor) -> Tensor:
        z_rev = grad_reverse(z)
        return super().forward(z_rev)


class Decoder(BaseModel):
    """Decoder."""

    def __init__(self, *, latent_dim: int, in_size: int, blocks: int, hid_multiplier: int) -> None:
        super().__init__(in_size=latent_dim, hid_size=latent_dim * hid_multiplier, out_size=in_size, blocks=blocks)


class Clf(CommonModel):
    """Main Autoencoder."""

    def __init__(
        self, cfg: ModelConfig, num_s: int, data_dim: int, s_dim: int, cf_available: bool, outcome_cols: List[str]
    ):
        super().__init__(name="Clf")
        self.enc = Encoder(
            in_size=data_dim + s_dim,
            latent_dim=cfg.latent_dims,
            blocks=cfg.blocks,
            hid_multiplier=cfg.latent_multiplier,
        )
        self.adv = Adversary(
            latent_dim=cfg.latent_dims, out_size=1, blocks=cfg.blocks, hid_multiplier=cfg.latent_multiplier
        )
        self.decoders = nn.ModuleList(
            [
                Decoder(latent_dim=cfg.latent_dims, in_size=1, blocks=0, hid_multiplier=cfg.latent_multiplier)
                for _ in range(num_s)
            ]
        )
        self.adv_weight = cfg.adv_weight
        self.reg_weight = cfg.reg_weight
        self.pred_weight = cfg.target_weight
        self.lr = cfg.lr
        self.s_input = cfg.s_as_input
        self.cf_model = cf_available
        self.ld = cfg.latent_dims
        self.mmd_kernel = cfg.mmd_kernel
        self.outcome_cols = outcome_cols
        self.scheduler_rate = cfg.scheduler_rate
        self.weight_decay = cfg.weight_decay

    @implements(nn.Module)
    def forward(self, x: Tensor, s: Tensor) -> Tuple[Tensor, Tensor, List[Tensor]]:
        _x = torch.cat([x, s[..., None]], dim=1) if self.s_input else x
        z = self.enc(_x)
        s_pred = self.adv(z)
        preds = [dec(z) for dec in self.decoders]
        return z, s_pred, preds

    @implements(LightningModule)
    def training_step(self, batch: Tuple[Tensor, ...], batch_idx: int) -> Tensor:
        if self.cf_model:
            x, s, y, cf_x, cf_s, cf_y, iw = batch
        else:
            x, s, y = batch
        z, s_pred, preds = self(x, s)
        pred_loss = binary_cross_entropy_with_logits(index_by_s(preds, s).squeeze(-1), y, reduction="mean", weight=iw)
        adv_loss = (
            mmd2(z[s == 0], z[s == 1], kernel=self.mmd_kernel)
            + binary_cross_entropy_with_logits(s_pred.squeeze(-1), s, reduction="mean")
        ) / 2
        loss = self.pred_weight * pred_loss + self.adv_weight * adv_loss

        to_log = {
            "training_clf/loss": loss,
            "training_clf/pred_loss": pred_loss,
            "training_clf/adv_loss": adv_loss,
            "training_clf/z_norm": z.detach().norm(dim=1).mean(),
            # "training_clf/z_dim_0": wandb.Histogram(z.detach().cpu().numpy()[:, 0]),
            # "training_clf/z_dim_0_s0": wandb.Histogram(z[s <= 0].detach().cpu().numpy()[:, 0]),
            # "training_clf/z_dim_0_s1": wandb.Histogram(z[s > 0].detach().cpu().numpy()[:, 0]),
            "training_clf/z_mean_abs_diff": (z[s <= 0].mean() - z[s > 0].mean()).abs(),
        }

        if self.cf_model:
            with no_grad():
                cf_z, _, cf_preds = self(cf_x, cf_s)
                cf_pred_loss = binary_cross_entropy_with_logits(
                    index_by_s(cf_preds, cf_s).squeeze(-1), cf_y, reduction="mean"
                )
                cf_loss = cf_pred_loss - 1e-6
                to_log["training_clf/cf_loss"] = cf_loss
                to_log["training_clf/cf_pred_loss"] = cf_pred_loss

        for k, v in to_log.items():
            do_log(k, v, self.logger)

        return loss

    @torch.no_grad()
    def threshold(self, z: Tensor) -> Tensor:
        """Go from soft to discrete features."""
        return z.sigmoid().round()

    @implements(LightningModule)
    def test_step(self, batch: Tuple[Tensor, ...], batch_idx: int) -> Dict[str, Tensor]:
        if self.cf_model:
            x, s, y, cf_x, cf_s, cf_y, _ = batch
        else:
            x, s, y = batch
        z, _, preds = self(x, s)

        to_return = {
            "y": y,
            "z": z,
            "s": s,
            "preds": self.threshold(index_by_s(preds, s)),
            "preds_0": self.threshold(preds[0]),
            "preds_1": self.threshold(preds[1]),
        }

        if self.cf_model:
            to_return["cf_y"] = cf_y
            to_return["cf_preds"] = self.threshold(index_by_s(preds, cf_s))

        return to_return

    @implements(LightningModule)
    def test_epoch_end(self, output_results: List[Dict[str, Tensor]]) -> None:
        all_y = torch.cat([_r["y"] for _r in output_results], 0)
        all_z = torch.cat([_r["z"] for _r in output_results], 0)
        all_s = torch.cat([_r["s"] for _r in output_results], 0)
        all_preds = torch.cat([_r["preds"] for _r in output_results], 0)
        torch.cat([_r["preds_0"] for _r in output_results], 0)
        torch.cat([_r["preds_1"] for _r in output_results], 0)

        make_plot(
            x=all_y.unsqueeze(-1),
            s=all_s,
            logger=self.logger,
            name="true_data",
            cols=self.outcome_cols,
        )
        make_plot(x=all_preds, s=all_s, logger=self.logger, name="preds", cols=self.outcome_cols)
        make_plot(x=all_z, s=all_s, logger=self.logger, name="z", cols=[str(i) for i in range(self.ld)])

        if self.cf_model:
            all_cf_y = torch.cat([_r["cf_y"] for _r in output_results], 0)
            cf_preds = torch.cat([_r["cf_preds"] for _r in output_results], 0)
            make_plot(
                x=all_cf_y.unsqueeze(-1),
                s=all_s,
                logger=self.logger,
                name="true_counterfactual_outcome",
                cols=self.outcome_cols,
            )
            make_plot(x=cf_preds, s=all_s, logger=self.logger, name="cf_preds", cols=self.outcome_cols)

    @implements(LightningModule)
    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[ExponentialLR]]:
        optimizer = Adam(self.parameters(), lr=self.lr, weight_decay=1e-6)
        scheduler = ExponentialLR(optimizer, gamma=self.scheduler_rate)
        return [optimizer], [scheduler]

    @implements(CommonModel)
    def get_recon(self, dataloader: DataLoader) -> np.ndarray:
        recons = None
        for batch in dataloader:
            if self.cf_model:
                x, s, y, cf_x, cf_s, cf_y, _ = batch
            else:
                x, s, y = batch

            x = x.to(self.device)
            s = s.to(self.device)
            _, _, _r = self(x, s)
            r = self.threshold(index_by_s(_r, s))
            recons = r if recons is None else cat([recons, r], dim=0)  # type: ignore[unreachable]
        assert recons is not None
        return recons.detach().cpu().numpy()

    def from_recons(self, recons: List[Tensor]) -> Dict[str, Tuple[Tensor, ...]]:
        """Given recons, give all possible predictions."""
        preds_dict: Dict[str, Tuple[Tensor, ...]] = {}

        for i, rec in enumerate(recons):
            z, s_pred, preds = self(rec, torch.ones_like(rec[:, 0]) * i)
            for _s in range(2):
                preds_dict[f"{i}_{_s}"] = (z, s_pred, preds[_s])
        return preds_dict
