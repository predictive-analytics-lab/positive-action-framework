from typing import Tuple, List, Dict

import torch
import wandb
from ethicml import implements
from pytorch_lightning import LightningModule
from torch import nn, Tensor, no_grad
from torch.nn.functional import l1_loss, binary_cross_entropy_with_logits
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from src.config_classes.dataclasses import ModelConfig
from src.mmd import mmd2
from src.model.blocks import block, mid_blocks
from src.model.model_utils import index_by_s, grad_reverse
from src.utils import make_plot


class BaseModel(nn.Module):
    def __init__(self, *, in_size: int, hid_size: int, out_size: int, blocks: int):
        super().__init__()
        if blocks == 0:
            self.hid = nn.Identity()
            self.out = nn.Linear(in_size, out_size)
        else:
            _blocks = [block(in_dim=in_size, out_dim=hid_size)] + mid_blocks(
                latent_dim=hid_size, blocks=blocks
            )
            self.hid = nn.Sequential(*_blocks)
            self.out = nn.Linear(hid_size, out_size)
        nn.init.xavier_normal_(self.out.weight)

    @implements(nn.Module)
    def forward(self, x: Tensor) -> Tensor:
        z = self.hid(x)
        return self.out(z)


class Encoder(BaseModel):
    @implements(BaseModel)
    def __init__(self, *, in_size: int, latent_dim: int, blocks: int, hid_multiplier: int):
        super().__init__(
            in_size=in_size,
            hid_size=latent_dim * hid_multiplier,
            out_size=latent_dim,
            blocks=blocks,
        )


class Adversary(BaseModel):
    @implements(BaseModel)
    def __init__(self, *, latent_dim: int, out_size: int, blocks: int, hid_multiplier: int):
        super().__init__(
            in_size=latent_dim,
            hid_size=latent_dim * hid_multiplier,
            out_size=out_size,
            blocks=blocks,
        )

    def forward(self, z: Tensor) -> Tensor:
        z_rev = grad_reverse(z)
        return super().forward(z_rev)


class Decoder(BaseModel):
    @implements(nn.Module)
    def __init__(self, *, latent_dim: int, in_size: int, blocks: int, hid_multiplier: int) -> None:
        super().__init__(
            in_size=latent_dim,
            hid_size=latent_dim * hid_multiplier,
            out_size=in_size,
            blocks=blocks,
        )


class AE(LightningModule):
    def __init__(self, cfg: ModelConfig, num_s: int, data_dim: int, s_dim: int):
        super().__init__()
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
                    in_size=1,
                    blocks=cfg.blocks,
                    hid_multiplier=cfg.latent_multiplier,
                )
                for _ in range(num_s)
            ]
        )
        self.reg_weight = cfg.reg_weight
        self.recon_weight = cfg.recon_weight
        self.lr = cfg.lr
        self.s_input = cfg.s_as_input

    def forward(self, x: Tensor, s: Tensor) -> Tuple[Tensor, Tensor, List[Tensor]]:
        _x = torch.cat([x, s[..., None]], dim=1) if self.s_input else x
        z = self.enc(_x)
        s_pred = self.adv(z)
        recons = [dec(z) for dec in self.decoders]
        return z, s_pred, recons

    def training_step(self, batch: Tuple[Tensor, ...], batch_idx: int) -> Tensor:
        x, s, y, cf_x, cf_s, cf_y = batch
        z, s_pred, recons = self(x, s)
        recon_loss = l1_loss(index_by_s(recons, s), x, reduction="mean")
        adv_loss = (
            mmd2(z[s == 0], z[s == 1], kernel="rbf")
            + binary_cross_entropy_with_logits(s_pred.squeeze(-1), s, reduction="mean")
        ) / 2
        loss = self.recon_weight * recon_loss + adv_loss

        with no_grad():
            cf_z, _, cf_recons = self(cf_x, cf_s)
            cf_recon_loss = l1_loss(index_by_s(cf_recons, cf_s), cf_x, reduction="mean")
            cf_loss = cf_recon_loss - 1e-6

            self.logger.experiment.log(
                {
                    "training/loss": loss,
                    "training/recon_loss": recon_loss,
                    "training/adv_loss": adv_loss,
                    "training/cf_loss": cf_loss,
                    "training/cf_recon_loss": cf_recon_loss,
                    "training/z_norm": z.norm(dim=1).mean(),
                    "training/z_dim_0": wandb.Histogram(z.cpu().numpy()[:, 0]),
                    "training/z_dim_0_s0": wandb.Histogram(z[s <= 0].cpu().numpy()[:, 0]),
                    "training/z_dim_0_s1": wandb.Histogram(z[s > 0].cpu().numpy()[:, 0]),
                    "training/z_mean_abs_diff": (z[s <= 0].mean() - z[s > 0].mean()).abs(),
                }
            )
        return loss

    def test_step(self, batch: Tuple[Tensor, ...], batch_idx: int) -> Dict[str, Tensor]:
        x, s, y, cf_x, cf_s, cf_y = batch
        z, _, recons = self(x, s)
        return {
            "x": x,
            "cf_x": cf_x,
            "z": z,
            "s": s,
            "recon": index_by_s(recons, s),
            "cf_recon": index_by_s(recons, cf_s),
            "recons_0": recons[0],
            "recons_1": recons[1],
        }

    def test_epoch_end(self, output_results: List[Dict[str, Tensor]]) -> None:
        all_x = torch.cat([_r["x"] for _r in output_results], 0)
        all_cf_x = torch.cat([_r["cf_x"] for _r in output_results], 0)
        all_z = torch.cat([_r["z"] for _r in output_results], 0)
        all_s = torch.cat([_r["s"] for _r in output_results], 0)
        all_recon = torch.cat([_r["recon"] for _r in output_results], 0)
        cf_recon = torch.cat([_r["cf_recon"] for _r in output_results], 0)
        recon_0 = torch.cat([_r["recons_0"] for _r in output_results], 0)
        recon_1 = torch.cat([_r["recons_1"] for _r in output_results], 0)

        make_plot(x=all_x, s=all_s, logger=self.logger, name="true_data")
        make_plot(x=all_cf_x, s=all_s, logger=self.logger, name="true_counterfactual")
        make_plot(x=all_recon, s=all_s, logger=self.logger, name="recons")
        make_plot(x=cf_recon, s=all_s, logger=self.logger, name="cf_recons")
        make_plot(x=recon_0, s=all_s, logger=self.logger, name="recons_all_s0")
        make_plot(x=recon_1, s=all_s, logger=self.logger, name="recons_all_s1")
        make_plot(x=all_z, s=all_s, logger=self.logger, name="z")

    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], ...]:
        optimizer = Adam(self.parameters(), lr=self.lr)
        scheduler = ExponentialLR(optimizer, gamma=0.99)
        return [optimizer], [scheduler]
