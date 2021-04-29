"""Encoder model."""
import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
from ethicml import implements
from pytorch_lightning import LightningModule
from torch import Tensor, cat, nn, no_grad
from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy, l1_loss, mse_loss
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR
from torch.utils.data import DataLoader

from paf.config_classes.dataclasses import KernelType
from paf.logging_i_guess import do_log
from paf.mmd import mmd2
from paf.model.blocks import block, mid_blocks
from paf.model.common_model import CommonModel
from paf.model.model_utils import grad_reverse, index_by_s, to_discrete
from paf.plotting import make_plot

logger = logging.getLogger(__name__)


class BaseModel(nn.Module):
    """Base AE Model."""

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
        s_as_input: bool,
        latent_dims: int,
        encoder_blocks: int,
        latent_multiplier: int,
        adv_blocks: int,
        decoder_blocks: int,
        adv_weight: float,
        reg_weight: float,
        target_weight: float,
        lr: float,
        mmd_kernel: KernelType,
        scheduler_rate: float,
        weight_decay: float,
        use_iw: bool,
    ):
        super().__init__(name="Enc")

        self.adv_weight = adv_weight
        self.reg_weight = reg_weight
        self.recon_weight = target_weight
        self.lr = lr
        self.s_as_input = s_as_input
        self.latent_dims = latent_dims
        self.mmd_kernel = mmd_kernel
        self.scheduler_rate = scheduler_rate
        self.weight_decay = weight_decay
        self.num_batches = 0
        self.encoder_blocks = encoder_blocks
        self.latent_multiplier = latent_multiplier
        self.adv_blocks = adv_blocks
        self.decoder_blocks = decoder_blocks
        self.built = False

    @implements(CommonModel)
    def build(
        self,
        num_s: int,
        data_dim: int,
        s_dim: int,
        cf_available: bool,
        feature_groups: Dict[str, List[slice]],
        outcome_cols: List[str],
    ):
        self.cf_model = cf_available
        self.feature_groups = feature_groups
        self.data_cols = outcome_cols

        self.enc = Encoder(
            in_size=data_dim + s_dim if self.s_as_input else data_dim,
            latent_dim=self.latent_dims,
            blocks=self.encoder_blocks,
            hid_multiplier=self.latent_multiplier,
        )
        self.adv = Adversary(
            latent_dim=self.latent_dims,
            out_size=1,
            blocks=self.adv_blocks,
            hid_multiplier=self.latent_multiplier,
        )
        self.decoders = nn.ModuleList(
            [
                Decoder(
                    latent_dim=self.latent_dims,
                    in_size=data_dim,
                    blocks=self.decoder_blocks,
                    hid_multiplier=self.latent_multiplier,
                )
                for _ in range(num_s)
            ]
        )
        self.built = True

    @implements(nn.Module)
    def forward(self, x: Tensor, s: Tensor) -> Tuple[Tensor, Tensor, List[Tensor]]:
        assert self.built
        _x = torch.cat([x, s[..., None]], dim=1) if self.s_as_input else x
        z = self.enc(_x)
        s_pred = self.adv(z)
        recons = [dec(z) for dec in self.decoders]
        return z, s_pred, recons

    @implements(LightningModule)
    def training_step(self, batch: Tuple[Tensor, ...], batch_idx: int) -> Tensor:
        assert self.built
        if batch_idx == 0:
            self.num_batches += 1
        if self.cf_model:
            x, s, _, cf_x, cf_s, _, _ = batch
        else:
            x, s, y, iw = batch
        z, s_pred, recons = self(x, s)

        if self.feature_groups["discrete"]:
            recon_loss = x.new_tensor(0.0)
            for i, feature_weight in zip(
                range(x[:, slice(self.feature_groups["discrete"][-1].stop, x.shape[1])].shape[1]),
                [0, 1e0, 1e0, 1e0, 1e0],
            ):
                recon_loss += (
                    mse_loss(
                        index_by_s(recons, s)[
                            :, slice(self.feature_groups["discrete"][-1].stop, x.shape[1])
                        ][:, i].sigmoid(),
                        x[:, slice(self.feature_groups["discrete"][-1].stop, x.shape[1])][:, i],
                    )
                    * feature_weight
                )
            # recon_loss = mse_loss(
            #     index_by_s(recons, s)[
            #         :, slice(self.feature_groups["discrete"][-1].stop, x.shape[1])
            #     ].sigmoid(),
            #     x[:, slice(self.feature_groups["discrete"][-1].stop, x.shape[1])],
            #     reduction="mean",
            # )
            _tmp_recon_loss = torch.zeros_like(recon_loss)
            for group_slice, feature_weight in zip(
                self.feature_groups["discrete"], [1e0, 1e0, 0, 1e0, 0, 1e0, 1e0]
            ):
                recon_loss += (
                    cross_entropy(
                        index_by_s(recons, s)[:, group_slice],
                        torch.argmax(x[:, group_slice], dim=-1),
                        reduction="mean",
                    )
                    * feature_weight
                )
            recon_loss += _tmp_recon_loss  # / len(self.feature_groups["discrete"])
        else:
            recon_loss = mse_loss(index_by_s(recons, s).sigmoid(), x, reduction="mean")

        if self.num_batches > 0:
            adv_loss = (
                mmd2(z[s == 0], z[s == 1], kernel=self.mmd_kernel)
                + binary_cross_entropy_with_logits(s_pred.squeeze(-1), s, reduction="mean")
            ) / 2
        else:
            adv_loss = torch.zeros_like(recon_loss)
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
    def invert(self, z: Tensor, x: Tensor) -> Tensor:
        """Go from soft to discrete features."""
        k = z.detach().clone()
        if self.feature_groups["discrete"]:
            for i in range(
                k[:, slice(self.feature_groups["discrete"][-1].stop, k.shape[1])].shape[1]
            ):
                if i == 0:
                    k[:, slice(self.feature_groups["discrete"][-1].stop, k.shape[1])][:, i] = x[
                        :, slice(self.feature_groups["discrete"][-1].stop, x.shape[1])
                    ][:, i]
                else:
                    k[:, slice(self.feature_groups["discrete"][-1].stop, k.shape[1])][:, i] = k[
                        :, slice(self.feature_groups["discrete"][-1].stop, k.shape[1])
                    ][:, i].sigmoid()
            for i, group_slice in enumerate(self.feature_groups["discrete"]):
                if i in [2, 4]:
                    k[:, group_slice] = x[:, group_slice]
                else:
                    one_hot = to_discrete(inputs=k[:, group_slice])
                    k[:, group_slice] = one_hot
        else:
            k = k.sigmoid()

        return k

    @implements(LightningModule)
    def test_step(self, batch: Tuple[Tensor, ...], batch_idx: int) -> Dict[str, Tensor]:
        assert self.built
        if self.cf_model:
            x, s, _, cf_x, cf_s, _, _ = batch
        else:
            x, s, _, _ = batch
        z, _, recons = self(x, s)

        to_return = {
            "x": x,
            "z": z,
            "s": s,
            "recon": self.invert(index_by_s(recons, s), x),
            "recons_0": self.invert(recons[0], x),
            "recons_1": self.invert(recons[1], x),
        }

        if self.cf_model:
            to_return["cf_x"] = cf_x
            to_return["cf_recon"] = self.invert(index_by_s(recons, cf_s), x)

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
        if self.feature_groups["discrete"]:
            make_plot(
                x=all_x[:, slice(self.feature_groups["discrete"][-1].stop, all_x.shape[1])],
                s=all_s,
                logger=self.logger,
                name="true_data",
                cols=self.data_cols[
                    slice(self.feature_groups["discrete"][-1].stop, all_x.shape[1])
                ],
            )
            make_plot(
                x=all_recon[:, slice(self.feature_groups["discrete"][-1].stop, all_x.shape[1])],
                s=all_s,
                logger=self.logger,
                name="recons",
                cols=self.data_cols[
                    slice(self.feature_groups["discrete"][-1].stop, all_x.shape[1])
                ],
            )
            for group_slice in self.feature_groups["discrete"]:
                make_plot(
                    x=all_x[:, group_slice],
                    s=all_s,
                    logger=self.logger,
                    name="true_data",
                    cols=self.data_cols[group_slice],
                    cat_plot=True,
                )
                make_plot(
                    x=all_recon[:, group_slice],
                    s=all_s,
                    logger=self.logger,
                    name="recons",
                    cols=self.data_cols[group_slice],
                    cat_plot=True,
                )
        else:
            make_plot(x=all_x, s=all_s, logger=self.logger, name="true_data", cols=self.data_cols)
            make_plot(x=all_recon, s=all_s, logger=self.logger, name="recons", cols=self.data_cols)
        make_plot(
            x=all_z,
            s=all_s,
            logger=self.logger,
            name="z",
            cols=[str(i) for i in range(self.latent_dims)],
        )
        recon_mse = (all_x - all_recon).mean(dim=0).abs()
        for i, feature_mse in enumerate(recon_mse):
            feature_name = self.data_cols[i]
            do_log(
                f"Table6/Ours/recon_l1 - feature {feature_name}",
                round(feature_mse.item(), 5),
                self.logger,
            )

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
            make_plot(
                x=cf_recon, s=all_s, logger=self.logger, name="cf_recons", cols=self.data_cols
            )

            recon_mse = (all_cf_x - cf_recon).mean(dim=0).abs()
            for i, feature_mse in enumerate(recon_mse):
                feature_name = self.data_cols[i]
                do_log(
                    f"Table6/Ours/cf_recon_l1 - feature {feature_name}",
                    round(feature_mse.item(), 5),
                    self.logger,
                )

    # @implements(LightningModule)
    # def validation_step(self, batch: Tuple[Tensor, ...], batch_idx: int) -> Dict[str, Tensor]:
    #
    #     if self.cf_model:
    #         x, s, _, cf_x, cf_s, _, _ = batch
    #     else:
    #         x, s, _ = batch
    #     z, s_pred, recons = self(x, s)
    #
    #     recon_loss = mse_loss(index_by_s(recons, s), x, reduction="mean")
    #     adv_loss = (
    #         mmd2(z[s == 0], z[s == 1], kernel=self.mmd_kernel)
    #         + binary_cross_entropy_with_logits(s_pred.squeeze(-1), s, reduction="mean")
    #     ) / 2
    #     loss = self.recon_weight * recon_loss + self.adv_weight * adv_loss
    #
    #     to_return = {"x": x, "z": z, "s": s, "recon": self.invert(index_by_s(recons, s)), "val_mse": loss}
    #
    #     if self.cf_model:
    #         to_return["cf_x"] = cf_x
    #         to_return["cf_recon"] = self.invert(index_by_s(recons, cf_s))
    #
    #     return to_return

    @implements(LightningModule)
    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[ExponentialLR]]:
        optimizer = Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        CosineAnnealingLR(optimizer, 10)
        # scheduler = ReduceLROnPlateau(optimizer, "min", patience=5)
        # ExponentialLR(optimizer, gamma=self.scheduler_rate)
        return optimizer
        # return [optimizer], [scheduler]
        # return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_mse'}

    @implements(CommonModel)
    def get_recon(self, dataloader: DataLoader) -> np.ndarray:
        recons = None
        for batch in dataloader:
            if self.cf_model:
                x, s, y, cf_x, cf_s, cf_y, _ = batch
            else:
                x, s, y, _ = batch
            x = x.to(self.device)
            s = s.to(self.device)
            _, _, _r = self(x, s)
            r = self.invert(index_by_s(_r, s), x)
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
                x, s, y, _ = batch
            x = x.to(self.device)
            s = s.to(self.device)
            _, _, _r = self(x, s)
            r0 = self.invert(_r[0], x)
            r1 = self.invert(_r[1], x)
            recons = torch.stack([r0, r1]) if recons is None else cat([recons, torch.stack([r0, r1])], dim=1)  # type: ignore[unreachable]
            sens = s if sens is None else cat([sens, s], dim=0)
            labels = y if labels is None else cat([labels, y], dim=0)
        assert recons is not None
        return recons.detach(), sens.detach(), labels.detach()
