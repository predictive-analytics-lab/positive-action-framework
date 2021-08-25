"""Encoder model."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import logging
from typing import NamedTuple

from bolts.structures import Stage
from kit import implements, parsable
import numpy as np
from pytorch_lightning import LightningModule
from sklearn.preprocessing import MinMaxScaler
import torch
from torch import Tensor, nn, no_grad, optim
from torch.nn.functional import (
    binary_cross_entropy_with_logits,
    cross_entropy,
    l1_loss,
    mse_loss,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchmetrics import MeanSquaredError

from paf.base_templates.dataset_utils import Batch, CfBatch
from paf.config_classes.dataclasses import KernelType
from paf.mmd import mmd2
from paf.model.blocks import block, mid_blocks
from paf.model.common_model import CommonModel
from paf.model.model_utils import grad_reverse, index_by_s, to_discrete
from paf.plotting import make_plot

logger = logging.getLogger(__name__)


@dataclass
class SharedStepOut:
    x: Tensor
    z: Tensor
    s: Tensor
    recon: Tensor
    cf_pred: Tensor
    recons_0: Tensor
    recons_1: Tensor


@dataclass
class CfSharedStepOut(SharedStepOut):
    cf_x: Tensor
    cf_recon: Tensor


class BaseModel(nn.Module):
    """Base AE Model."""

    hid: nn.Module
    out: nn.Module

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


class Loss:
    def __init__(
        self,
        loss_type: str = 'MSE',
        feature_groups: dict[str, list[slice]] | None = None,
        adv_weight: float = 1.0,
        cycle_weight: float = 1.0,
        recon_weight: float = 1.0,
    ):
        self._recon_loss_fn = nn.L1Loss(reduction="mean")
        self.feature_groups = feature_groups if feature_groups is not None else {}
        self._adv_weight = adv_weight
        self._cycle_weight = cycle_weight
        self._recon_weight = recon_weight
        self._cycle_loss_fn = nn.MSELoss(reduction="mean")

    def recon_loss(self, recons: list[Tensor], batch: Batch | CfBatch) -> Tensor:

        if self.feature_groups["discrete"]:
            recon_loss = batch.x.new_tensor(0.0)
            for i in range(
                batch.x[:, slice(self.feature_groups["discrete"][-1].stop, batch.x.shape[1])].shape[
                    1
                ]
            ):
                recon_loss += self._recon_loss_fn(
                    index_by_s(recons, batch.s)[
                        :, slice(self.feature_groups["discrete"][-1].stop, batch.x.shape[1])
                    ][:, i].sigmoid(),
                    batch.x[:, slice(self.feature_groups["discrete"][-1].stop, batch.x.shape[1])][
                        :, i
                    ],
                )
            for group_slice in self.feature_groups["discrete"]:
                recon_loss += cross_entropy(
                    index_by_s(recons, batch.s)[:, group_slice],
                    torch.argmax(batch.x[:, group_slice], dim=-1),
                    reduction="mean",
                )
        else:
            recon_loss = self._recon_loss_fn(
                index_by_s(recons, batch.s).sigmoid(), batch.x, reduction="mean"
            )

        return recon_loss * self._recon_weight

    def adv_loss(self, enc_fwd: EncFwd, batch: Batch | CfBatch, kernel: KernelType) -> Tensor:
        return (
            (
                mmd2(enc_fwd.z[batch.s == 0], enc_fwd.z[batch.s == 1], kernel=kernel)
                + binary_cross_entropy_with_logits(enc_fwd.s.squeeze(-1), batch.s, reduction="mean")
            )
            / 2
        ) * self._adv_weight

    def cycle_loss(self, cyc_fwd: EncFwd, batch: Batch | CfBatch) -> Tensor:
        return (
            self._cycle_loss_fn(index_by_s(cyc_fwd.x, batch.s).squeeze(-1), batch.x)
            * self._cycle_weight
        )


class AE(CommonModel):
    """Main Autoencoder."""

    feature_groups: dict[str, list[slice]]
    cf_model: bool
    data_cols: list[str]

    @parsable
    def __init__(
        self,
        s_as_input: bool,
        latent_dims: int,
        encoder_blocks: int,
        latent_multiplier: int,
        adv_blocks: int,
        decoder_blocks: int,
        adv_weight: float,
        cycle_weight: float,
        target_weight: float,
        lr: float,
        mmd_kernel: KernelType,
        scheduler_rate: float,
        weight_decay: float,
        use_iw: bool,
    ):
        super().__init__(name="Enc")

        self._adv_weight = adv_weight
        self._cycle_weight = cycle_weight
        self._recon_weight = target_weight
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

        self.val_mse = MeanSquaredError()
        self.test_mse = MeanSquaredError()

    @implements(CommonModel)
    def build(
        self,
        num_s: int,
        data_dim: int,
        s_dim: int,
        cf_available: bool,
        feature_groups: dict[str, list[slice]],
        outcome_cols: list[str],
        scaler: MinMaxScaler,
    ) -> None:
        self.cf_model = cf_available
        self.data_cols = outcome_cols
        self.scaler = scaler

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
        self.loss = Loss(
            feature_groups=feature_groups,
            adv_weight=self._adv_weight,
            cycle_weight=self._cycle_weight,
            recon_weight=self._recon_weight,
        )
        self.built = True

    @implements(nn.Module)
    def forward(self, x: Tensor, s: Tensor) -> EncFwd:
        assert self.built
        _x = torch.cat([x, s[..., None]], dim=1) if self.s_as_input else x
        z = self.enc.forward(_x)
        s_pred = self.adv.forward(z)
        recons = [dec(z) for dec in self.decoders]
        return EncFwd(z=z, s=s_pred, x=recons)

    @implements(LightningModule)
    def training_step(self, batch: Batch | CfBatch, batch_idx: int) -> Tensor:
        assert self.built
        enc_fwd = self.forward(batch.x, batch.s)

        recon_loss = self.loss.recon_loss(enc_fwd.x, batch)
        adv_loss = self.loss.adv_loss(enc_fwd, batch, self.mmd_kernel)
        cyc_fwd = self.forward(index_by_s(enc_fwd.x, 1 - batch.s), 1 - batch.s)
        cycle_loss = self.loss.cycle_loss(cyc_fwd, batch)

        loss = recon_loss + adv_loss + cycle_loss

        to_log = {
            "training_enc/loss": loss,
            "training_enc/recon_loss": recon_loss,
            "training_enc/adv_loss": adv_loss,
            "training_enc/z_norm": enc_fwd.z.detach().norm(dim=1).mean(),
            "training_enc/z_mean_abs_diff": (
                enc_fwd.z[batch.s <= 0].mean() - enc_fwd.z[batch.s > 0].mean()
            ).abs(),
        }

        if isinstance(batch, CfBatch):
            with no_grad():
                _, _, cf_recons = self.forward(batch.cfx, batch.cfs)
                cf_recon_loss = l1_loss(
                    index_by_s(cf_recons, batch.cfs), batch.cfx, reduction="mean"
                )
                cf_loss = cf_recon_loss - 1e-6
                to_log["training_enc/cf_loss"] = cf_loss
                to_log["training_enc/cf_recon_loss"] = cf_recon_loss

        self.log_dict(to_log, logger=True)

        return loss

    @torch.no_grad()
    def invert(self, z: Tensor, x: Tensor) -> Tensor:
        """Go from soft to discrete features."""
        k = z.detach().clone()
        if self.loss.feature_groups["discrete"]:
            for i in range(
                k[:, slice(self.loss.feature_groups["discrete"][-1].stop, k.shape[1])].shape[1]
            ):
                if i in []:  # [0]: Features to transplant to the reconstrcution
                    k[:, slice(self.loss.feature_groups["discrete"][-1].stop, k.shape[1])][
                        :, i
                    ] = x[:, slice(self.loss.feature_groups["discrete"][-1].stop, x.shape[1])][:, i]
                else:
                    k[:, slice(self.loss.feature_groups["discrete"][-1].stop, k.shape[1])][
                        :, i
                    ] = k[:, slice(self.loss.feature_groups["discrete"][-1].stop, k.shape[1])][
                        :, i
                    ].sigmoid()
            for i, group_slice in enumerate(self.loss.feature_groups["discrete"]):
                if i in []:  # [2, 4]: Features to transplant
                    k[:, group_slice] = x[:, group_slice]
                else:
                    one_hot = to_discrete(inputs=k[:, group_slice])
                    k[:, group_slice] = one_hot
        else:
            k = k.sigmoid()

        return k

    @implements(LightningModule)
    def test_step(self, batch: Batch | CfBatch, batch_idx: int) -> SharedStepOut | CfSharedStepOut:
        return self.shared_step(batch, batch_idx, Stage.test)

    @implements(LightningModule)
    def validation_step(
        self, batch: Batch | CfBatch, batch_idx: int
    ) -> SharedStepOut | CfSharedStepOut:
        return self.shared_step(batch, batch_idx, Stage.validate)

    def shared_step(
        self, batch: Batch | CfBatch, batch_idx: int, stage: Stage
    ) -> SharedStepOut | CfSharedStepOut:
        assert self.built
        enc_fwd = self.forward(batch.x, batch.s)

        to_return = SharedStepOut(
            x=batch.x,
            z=enc_fwd.z,
            s=batch.s,
            recon=self.invert(index_by_s(enc_fwd.x, batch.s), batch.x),
            cf_pred=self.invert(index_by_s(enc_fwd.x, 1 - batch.s), batch.x),
            recons_0=self.invert(enc_fwd.x[0], batch.x),
            recons_1=self.invert(enc_fwd.x[1], batch.x),
        )

        if isinstance(batch, CfBatch):
            to_return = CfSharedStepOut(
                cf_x=batch.cfx,
                cf_recon=self.invert(index_by_s(enc_fwd.x, batch.cfs), batch.x),
                **asdict(to_return),
            )

        return to_return

    @implements(LightningModule)
    def test_epoch_end(self, output_results: list[SharedStepOut | CfSharedStepOut]) -> None:
        self.shared_epoch_end(output_results, stage=Stage.test)

    @implements(LightningModule)
    def validation_epoch_end(self, output_results: list[SharedStepOut | CfSharedStepOut]) -> None:
        self.shared_epoch_end(output_results, stage=Stage.validate)

    def shared_epoch_end(
        self, output_results: list[SharedStepOut | CfSharedStepOut], stage: Stage
    ) -> None:
        self.all_x = torch.cat([_r.x for _r in output_results], 0)
        all_z = torch.cat([_r.z for _r in output_results], 0)
        self.all_s = torch.cat([_r.s for _r in output_results], 0)
        self.all_recon = torch.cat([_r.recon for _r in output_results], 0)
        self.all_cf_pred = torch.cat([_r.cf_pred for _r in output_results], 0)

        make_plot(
            x=all_z.clone(),
            s=self.all_s.clone(),
            logger=self.logger,
            name=f"{stage.value}_z",
            cols=[str(i) for i in range(self.latent_dims)],
        )

        if isinstance(output_results[0], CfSharedStepOut):
            all_cf_x = torch.cat([_r.cf_x for _r in output_results], 0)  # type: ignore[union-attr]
            cf_recon = torch.cat([_r.cf_recon for _r in output_results], 0)  # type: ignore[union-attr]
            make_plot(
                x=all_cf_x.clone(),
                s=self.all_s.clone(),
                logger=self.logger,
                name=f"{stage.value}_true_counterfactual",
                cols=self.data_cols,
            )
            make_plot(
                x=cf_recon.clone(),
                s=self.all_s.clone(),
                logger=self.logger,
                name=f"{stage.value}_cf_recons",
                cols=self.data_cols,
            )

    @implements(LightningModule)
    def configure_optimizers(
        self,
    ) -> tuple[list[optim.Optimizer], list[optim.lr_scheduler.ExponentialLR]]:
        opt = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = optim.lr_scheduler.ExponentialLR(opt, gamma=0.999)
        return [opt], [sched]

    @implements(CommonModel)
    def get_recon(self, dataloader: DataLoader) -> np.ndarray:
        recons: list[Tensor] | None = None
        for batch in dataloader:
            x = batch.x.to(self.device)
            s = batch.s.to(self.device)
            enc_fwd = self.forward(x, s)
            r = self.invert(index_by_s(enc_fwd.x, s), x)
            if recons is None:
                recons = [r]
            else:
                recons.append(r)
        assert recons is not None
        _recons = torch.cat(recons, dim=0)
        return _recons.detach().cpu().numpy()

    def run_through(self, dataloader: DataLoader) -> tuple[Tensor, Tensor, Tensor]:
        """Run through a dataloader and record the outputs with labels."""
        recons: list[Tensor] | None = None
        sens: list[Tensor] | None = None
        labels: list[Tensor] | None = None
        for batch in dataloader:
            x = batch.x.to(self.device)
            s = batch.s.to(self.device)
            y = batch.y.to(self.device)
            enc_fwd = self.forward(x, s)
            r0 = self.invert(enc_fwd.x[0], x)
            r1 = self.invert(enc_fwd.x[1], x)
            if recons is None:
                recons = [torch.stack([r0, r1])]
            else:
                recons.append(torch.stack([r0, r1]))
            if sens is None:
                sens = [s]
            else:
                sens.append(s)
            if labels is None:
                labels = [y]
            else:
                labels.append(y)
        assert recons is not None
        _recons = torch.cat(recons, dim=0)
        assert sens is not None
        _sens = torch.cat(sens, dim=0)
        assert labels is not None
        _labels = torch.cat(labels, dim=0)
        return RunThroughOut(x=_recons.detach(), s=_sens.detach(), y=_labels.detach())


class RunThroughOut(NamedTuple):
    x: Tensor
    s: Tensor
    y: Tensor


class EncFwd(NamedTuple):
    z: Tensor
    s: Tensor
    x: list[Tensor]
