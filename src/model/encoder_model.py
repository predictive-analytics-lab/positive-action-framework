"""Encoder model."""
import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
from ethicml import implements
from pytorch_lightning import LightningModule
from torch import Tensor, cat, nn, no_grad
from torch.nn.functional import binary_cross_entropy_with_logits, l1_loss, mse_loss
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from src.config_classes.dataclasses import ModelConfig
from src.mmd import mmd2
from src.model.blocks import block, mid_blocks
from src.model.common_model import CommonModel
from src.model.model_utils import grad_reverse, index_by_s, to_discrete
from src.utils import do_log, make_plot

logger = logging.getLogger(__name__)


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
        # nn.init.xavier_normal_(self.out.weight)

    @implements(nn.Module)
    def forward(self, x: Tensor) -> Tensor:
        z = self.hid(x)
        return self.out(z)


class Encoder(BaseModel):
    """AE Shared Encoder."""

    def __init__(self, *, in_size: int, latent_dim: int, blocks: int, hid_multiplier: int):
        super().__init__(
            in_size=in_size,
            hid_size=latent_dim * hid_multiplier,
            out_size=latent_dim,
            blocks=blocks,
        )


class Adversary(BaseModel):
    """AE Adversary head."""

    def __init__(self, *, latent_dim: int, out_size: int, blocks: int, hid_multiplier: int):
        super().__init__(
            in_size=latent_dim,
            hid_size=latent_dim * hid_multiplier,
            out_size=out_size,
            blocks=blocks,
        )

    @implements(nn.Module)
    def forward(self, z: Tensor) -> Tensor:
        z_rev = grad_reverse(z)
        return super().forward(z_rev)


class Decoder(BaseModel):
    """Decoder."""

    def __init__(self, *, latent_dim: int, in_size: int, blocks: int, hid_multiplier: int) -> None:
        super().__init__(
            in_size=latent_dim,
            hid_size=latent_dim * hid_multiplier,
            out_size=in_size,
            blocks=blocks,
        )


class AE(CommonModel):
    """Main Autoencoder."""

    def __init__(
        self,
        cfg: ModelConfig,
        num_s: int,
        data_dim: int,
        s_dim: int,
        cf_available: bool,
        feature_groups: Dict[str, List[slice]],
        column_names: List[str],
    ):
        super().__init__(name="Enc")
        self.enc = Encoder(
            in_size=data_dim + s_dim if cfg.s_as_input else data_dim,
            latent_dim=cfg.latent_dims,
            blocks=cfg.blocks,
            hid_multiplier=cfg.latent_multiplier,
        )
        self.adv = Adversary(
            latent_dim=cfg.latent_dims,
            out_size=1,
            blocks=cfg.blocks,
            hid_multiplier=cfg.latent_multiplier,
        )
        self.decoders = nn.ModuleList(
            [
                Decoder(
                    latent_dim=cfg.latent_dims,
                    in_size=data_dim,
                    blocks=cfg.blocks,
                    hid_multiplier=cfg.latent_multiplier,
                )
                for _ in range(num_s)
            ]
        )
        self.feature_groups = feature_groups
        self.adv_weight = cfg.adv_weight
        self.reg_weight = cfg.reg_weight
        self.recon_weight = cfg.target_weight
        self.lr = cfg.lr
        self.s_input = cfg.s_as_input
        self.cf_model = cf_available
        self.data_cols = column_names
        self.ld = cfg.latent_dims
        self.mmd_kernel = cfg.mmd_kernel
        self.scheduler_rate = cfg.scheduler_rate
        self.weight_decay = cfg.weight_decay

    @implements(nn.Module)
    def forward(self, x: Tensor, s: Tensor) -> Tuple[Tensor, Tensor, List[Tensor]]:
        _x = torch.cat([x, s[..., None]], dim=1) if self.s_input else x
        z = self.enc(_x)
        s_pred = self.adv(z)
        recons = [dec(z).sigmoid() for dec in self.decoders]
        return z, s_pred, recons

    @implements(LightningModule)
    def training_step(self, batch: Tuple[Tensor, ...], batch_idx: int) -> Tensor:
        if self.cf_model:
            x, s, _, cf_x, cf_s, _, _ = batch
        else:
            x, s, y, _ = batch
        z, s_pred, recons = self(x, s)
        recon_loss = mse_loss(index_by_s(recons, s), x, reduction="mean")
        adv_loss = (
            mmd2(z[s == 0], z[s == 1], kernel=self.mmd_kernel)
            + binary_cross_entropy_with_logits(s_pred.squeeze(-1), s, reduction="mean")
        ) / 2
        loss = self.recon_weight * recon_loss + self.adv_weight * adv_loss

        to_log = {
            "training_enc/loss": loss,
            "training_enc/recon_loss": recon_loss,
            "training_enc/adv_loss": adv_loss,
            "training_enc/z_norm": z.detach().norm(dim=1).mean(),
            # "training_enc/z_dim_0": wandb.Histogram(z.detach().cpu().numpy()[:, 0]),
            # "training_enc/z_dim_0_s0": wandb.Histogram(z[s <= 0].detach().cpu().numpy()[:, 0]),
            # "training_enc/z_dim_0_s1": wandb.Histogram(z[s > 0].detach().cpu().numpy()[:, 0]),
            "training_enc/z_mean_abs_diff": (z[s <= 0].mean() - z[s > 0].mean()).abs(),
        }

        if self.cf_model:
            with no_grad():
                _, _, cf_recons = self(cf_x, cf_s)
                cf_recon_loss = l1_loss(index_by_s(cf_recons, cf_s), cf_x, reduction="mean")
                cf_loss = cf_recon_loss - 1e-6
                to_log["training_enc/cf_loss"] = cf_loss
                to_log["training_enc/cf_recon_loss"] = cf_recon_loss

        for k, v in to_log.items():
            do_log(k, v, self.logger)
        return loss

    @torch.no_grad()
    def invert(self, z: Tensor) -> Tensor:
        """Go from soft to discrete features."""
        if self.feature_groups["discrete"]:
            for group_slice in self.feature_groups["discrete"]:
                one_hot = to_discrete(inputs=z[:, group_slice])
                z[:, group_slice] = one_hot

        return z

    @implements(LightningModule)
    def test_step(self, batch: Tuple[Tensor, ...], batch_idx: int) -> Dict[str, Tensor]:
        if self.cf_model:
            x, s, _, cf_x, cf_s, _, _ = batch
        else:
            x, s, _ = batch
        z, _, recons = self(x, s)

        to_return = {
            "x": x,
            "z": z,
            "s": s,
            "recon": self.invert(index_by_s(recons, s)),
            "recons_0": self.invert(recons[0]),
            "recons_1": self.invert(recons[1]),
        }

        if self.cf_model:
            to_return["cf_x"] = cf_x
            to_return["cf_recon"] = self.invert(index_by_s(recons, cf_s))

        return to_return

    @implements(LightningModule)
    def test_epoch_end(self, output_results: List[Dict[str, Tensor]]) -> None:
        all_x = torch.cat([_r["x"] for _r in output_results], 0)
        all_z = torch.cat([_r["z"] for _r in output_results], 0)
        all_s = torch.cat([_r["s"] for _r in output_results], 0)
        all_recon = torch.cat([_r["recon"] for _r in output_results], 0)
        torch.cat([_r["recons_0"] for _r in output_results], 0)
        torch.cat([_r["recons_1"] for _r in output_results], 0)

        logger.info(self.data_cols)
        make_plot(x=all_x, s=all_s, logger=self.logger, name="true_data", cols=self.data_cols)
        make_plot(x=all_recon, s=all_s, logger=self.logger, name="recons", cols=self.data_cols)
        make_plot(x=all_z, s=all_s, logger=self.logger, name="z", cols=[str(i) for i in range(self.ld)])
        recon_mse = (all_x - all_recon).mean(dim=0).abs()
        for i, feature_mse in enumerate(recon_mse):
            feature_name = self.data_cols[i]
            do_log(f"Table6/Ours/recon_l1 - feature {feature_name}", round(feature_mse.item(), 5), self.logger)

        if self.cf_model:
            all_cf_x = torch.cat([_r["cf_x"] for _r in output_results], 0)
            cf_recon = torch.cat([_r["cf_recon"] for _r in output_results], 0)
            make_plot(
                x=all_cf_x,
                s=all_s,
                logger=self.logger,
                name="true_counterfactual",
                cols=self.data_cols,
            )
            make_plot(x=cf_recon, s=all_s, logger=self.logger, name="cf_recons", cols=self.data_cols)

            recon_mse = (all_cf_x - cf_recon).mean(dim=0).abs()
            for i, feature_mse in enumerate(recon_mse):
                feature_name = self.data_cols[i]
                do_log(f"Table6/Ours/cf_recon_l1 - feature {feature_name}", round(feature_mse.item(), 5), self.logger)

    @implements(LightningModule)
    def validation_step(self, batch: Tuple[Tensor, ...], batch_idx: int) -> Dict[str, Tensor]:
        if self.cf_model:
            x, s, _, cf_x, cf_s, _, _ = batch
        else:
            x, s, _ = batch
        z, s_pred, recons = self(x, s)

        recon_loss = mse_loss(index_by_s(recons, s), x, reduction="mean")
        adv_loss = (
            mmd2(z[s == 0], z[s == 1], kernel=self.mmd_kernel)
            + binary_cross_entropy_with_logits(s_pred.squeeze(-1), s, reduction="mean")
        ) / 2
        loss = self.recon_weight * recon_loss + self.adv_weight * adv_loss

        to_return = {"x": x, "z": z, "s": s, "recon": self.invert(index_by_s(recons, s)), "val_mse": loss}

        if self.cf_model:
            to_return["cf_x"] = cf_x
            to_return["cf_recon"] = self.invert(index_by_s(recons, cf_s))

        return to_return

    @implements(LightningModule)
    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[ExponentialLR]]:
        optimizer = Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
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
            r = self.invert(index_by_s(_r, s))
            recons = r if recons is None else cat([recons, r], dim=0)  # type: ignore[unreachable]
        assert recons is not None
        return recons.detach().cpu().numpy()

    def run_through(self, dataloader: DataLoader) -> Tensor:
        """Run through a dataloader and record the outputs with labels."""
        recons = None
        sens = None
        labels = None
        for batch in dataloader:
            if self.cf_model:
                x, s, y, cf_x, cf_s, cf_y, _ = batch
            else:
                x, s, y = batch
            x = x.to(self.device)
            s = s.to(self.device)
            _, _, _r = self(x, s)
            r0 = self.invert(_r[0])
            r1 = self.invert(_r[1])
            recons = torch.stack([r0, r1]) if recons is None else cat([recons, torch.stack([r0, r1])], dim=1)  # type: ignore[unreachable]
            sens = s if sens is None else cat([sens, s], dim=0)
            labels = y if labels is None else cat([labels, y], dim=0)
        assert recons is not None
        return recons.detach(), sens.detach(), labels.detach()
