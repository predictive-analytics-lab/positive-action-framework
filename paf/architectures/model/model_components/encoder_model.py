"""Encoder model."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import logging
from typing import Any, NamedTuple

from conduit.data import TernarySample
from conduit.types import Stage
import numpy as np
import pytorch_lightning as pl
from ranzen import implements, parsable
from sklearn.preprocessing import MinMaxScaler
import torch
from torch import Tensor, nn, no_grad, optim
from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy, l1_loss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchmetrics import MeanSquaredError

from paf.base_templates.dataset_utils import Batch, CfBatch
from paf.mmd import KernelType, mmd2
from paf.plotting import make_plot

from .common_model import Adversary, BaseModel, CommonModel, Decoder, Encoder
from .model_utils import index_by_s

__all__ = [
    "SharedStepOut",
    "CfSharedStepOut",
    "BaseModel",
    "Loss",
    "AE",
    "RunThroughOut",
    "EncFwd",
]

from ... import MmdReportingResults

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


class RunThroughOut(NamedTuple):
    x: Tensor
    s: Tensor
    y: Tensor


class EncFwd(NamedTuple):
    z: Tensor
    s: Tensor
    x: list[Tensor]
    cyc_x: list[Tensor]
    cyc_z: Tensor


class Loss:
    def __init__(
        self,
        *,
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

    def recon_loss(self, recons: list[Tensor], *, batch: Batch | CfBatch | TernarySample) -> Tensor:

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
            recon_loss = self._recon_loss_fn(index_by_s(recons, batch.s).sigmoid(), batch.x)

        return recon_loss * self._recon_weight

    def adv_loss(self, enc_fwd: EncFwd, *, batch: Batch | CfBatch | TernarySample) -> Tensor:
        return (
            binary_cross_entropy_with_logits(enc_fwd.s.squeeze(-1), batch.s, reduction="mean")
        ) * self._adv_weight

    def mmd_loss(
        self, enc_fwd: EncFwd, *, batch: Batch | CfBatch | TernarySample, kernel: KernelType
    ) -> Tensor:
        return (
            mmd2(enc_fwd.z[batch.s == 0], enc_fwd.z[batch.s == 1], kernel=kernel) * self._adv_weight
        )

    def cycle_loss(
        self, cyc_x: list[Tensor], *, batch: Batch | CfBatch | TernarySample
    ) -> tuple[Tensor, Tensor]:
        cycle_loss = self._cycle_loss_fn(index_by_s(cyc_x, batch.s).squeeze(-1), batch.x)
        return cycle_loss, cycle_loss * self._cycle_weight


class AE(CommonModel):
    """Main Autoencoder."""

    cf_model: bool
    data_cols: list[str]
    scaler: MinMaxScaler
    enc: Encoder
    adv: Adversary
    all_x: Tensor
    all_s: Tensor
    all_recon: Tensor
    all_cf_pred: Tensor
    decoders: nn.ModuleList
    loss: Loss

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
        debug: bool,
    ):
        super().__init__(name="Enc")

        self._adv_weight = adv_weight
        self._cycle_weight = cycle_weight
        self._recon_weight = target_weight
        self.learning_rate = lr
        self.s_as_input = s_as_input
        self.latent_dims = latent_dims
        self.mmd_kernel = mmd_kernel
        self.scheduler_rate = scheduler_rate
        self.weight_decay = weight_decay
        self.encoder_blocks = encoder_blocks
        self.latent_multiplier = latent_multiplier
        self.adv_blocks = adv_blocks
        self.decoder_blocks = decoder_blocks
        self.debug = debug
        self.built = False

        self.val_mse = MeanSquaredError()
        self.test_mse = MeanSquaredError()

    @implements(CommonModel)
    def build(
        self,
        *,
        num_s: int,
        data_dim: int,
        s_dim: int,
        cf_available: bool,
        feature_groups: dict[str, list[slice]],
        outcome_cols: list[str],
        scaler: MinMaxScaler,
    ) -> None:
        _ = (scaler, cf_available)
        self.data_cols = outcome_cols

        self.adv = Adversary(
            latent_dim=self.latent_dims,
            out_size=1,
            blocks=self.adv_blocks,
            hid_multiplier=self.latent_multiplier,
        )
        self.enc = Encoder(
            in_size=data_dim + s_dim if self.s_as_input else data_dim,
            latent_dim=self.latent_dims,
            blocks=self.encoder_blocks,
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
    def forward(self, x: Tensor, *, s: Tensor) -> EncFwd:
        assert self.built
        _x = torch.cat([x, s[..., None]], dim=1) if self.s_as_input else x
        z = self.enc.forward(_x)
        s_pred = self.adv.forward(z)
        recons = [dec(z) for dec in self.decoders]

        cycle_x = (
            torch.cat([index_by_s(recons, 1 - s), 1 - s[..., None]], dim=1)
            if self.s_as_input
            else index_by_s(recons, 1 - s)
        )
        cycle_z = self.enc.forward(cycle_x)
        cycle_dec = [dec(cycle_z) for dec in self.decoders]
        return EncFwd(z=z, s=s_pred, x=recons, cyc_z=cycle_z, cyc_x=cycle_dec)

    @implements(pl.LightningModule)
    def training_step(self, batch: Batch | CfBatch | TernarySample, *_: Any) -> Tensor:
        assert self.built
        enc_fwd = self.forward(x=batch.x, s=batch.s)

        recon_loss = self.loss.recon_loss(recons=enc_fwd.x, batch=batch)
        adv_loss = self.loss.adv_loss(enc_fwd=enc_fwd, batch=batch)
        mmd_loss = self.loss.mmd_loss(enc_fwd=enc_fwd, batch=batch, kernel=self.mmd_kernel)
        report_of_cyc_loss, cycle_loss = self.loss.cycle_loss(cyc_x=enc_fwd.cyc_x, batch=batch)

        loss = recon_loss + adv_loss + cycle_loss + mmd_loss

        mmd_results = self.mmd_reporting(enc_fwd=enc_fwd, batch=batch)

        to_log = {
            f"{Stage.fit}/enc/loss": loss,
            f"{Stage.fit}/enc/recon_loss": recon_loss,
            f"{Stage.fit}/enc/mmd_loss": mmd_loss,
            f"{Stage.fit}/enc/adv_loss": adv_loss,
            f"{Stage.fit}/enc/z_norm": enc_fwd.z.detach().norm(dim=1).mean(),
            f"{Stage.fit}/enc/z_mean_abs_diff": (
                enc_fwd.z[batch.s <= 0].mean() - enc_fwd.z[batch.s > 0].mean()
            ).abs(),
            f"{Stage.fit}/enc/cycle_loss": report_of_cyc_loss,
            f"{Stage.fit}/enc/recon_mmd": mmd_results.recon,
            f"{Stage.fit}/enc/cf_recon_mmd": mmd_results.cf_recon,
            f"{Stage.fit}/enc/s0_dist_mmd": mmd_results.s0_dist,
            f"{Stage.fit}/enc/s1_dist_mmd": mmd_results.s1_dist,
        }

        if isinstance(batch, CfBatch):
            with no_grad():
                enc_fwd = self.forward(x=batch.cfx, s=batch.cfs)
                cf_recon_loss = l1_loss(
                    index_by_s(enc_fwd.x, batch.cfs), batch.cfx, reduction="mean"
                )
                cf_loss = cf_recon_loss - 1e-6
                to_log[f"{Stage.fit}/enc/cf_loss"] = cf_loss
                to_log[f"{Stage.fit}/enc/cf_recon_loss"] = cf_recon_loss

        self.log_dict(to_log, logger=True)

        return loss

    @implements(pl.LightningModule)
    def test_step(
        self, batch: Batch | CfBatch | TernarySample, *_: Any
    ) -> SharedStepOut | CfSharedStepOut:
        return self.shared_step(batch, stage=Stage.test)

    @implements(pl.LightningModule)
    def validation_step(
        self, batch: Batch | CfBatch | TernarySample, *_: Any
    ) -> SharedStepOut | CfSharedStepOut:
        return self.shared_step(batch, stage=Stage.validate)

    def shared_step(
        self, batch: Batch | CfBatch | TernarySample, *, stage: Stage
    ) -> SharedStepOut | CfSharedStepOut:
        assert self.built
        enc_fwd = self.forward(x=batch.x, s=batch.s)

        recon_loss = self.loss.recon_loss(recons=enc_fwd.x, batch=batch)
        adv_loss = self.loss.adv_loss(enc_fwd=enc_fwd, batch=batch)
        mmd_loss = self.loss.mmd_loss(enc_fwd=enc_fwd, batch=batch, kernel=self.mmd_kernel)
        cycle_loss, _ = self.loss.cycle_loss(cyc_x=enc_fwd.cyc_x, batch=batch)

        loss = recon_loss + adv_loss + cycle_loss + mmd_loss

        mmd_results = self.mmd_reporting(enc_fwd=enc_fwd, batch=batch)

        self.log(f"{stage}/enc/loss", loss)
        self.log(f"{stage}/enc/recon_loss", recon_loss)
        self.log(f"{stage}/enc/cycle_loss", cycle_loss)
        self.log(f"{stage}/enc/mmd_loss", mmd_loss)
        self.log(f"{stage}/enc/recon_mmd", mmd_results.recon)
        self.log(f"{stage}/enc/cf_recon_mmd", mmd_results.cf_recon)
        self.log(f"{stage}/enc/s0_dist_mmd", mmd_results.s0_dist)
        self.log(f"{stage}/enc/s1_dist_mmd", mmd_results.s1_dist)

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

    @implements(pl.LightningModule)
    def test_epoch_end(self, outputs: list[SharedStepOut | CfSharedStepOut]) -> None:
        self.shared_epoch_end(outputs, stage=Stage.test)

    @implements(pl.LightningModule)
    def validation_epoch_end(self, outputs: list[SharedStepOut | CfSharedStepOut]) -> None:
        self.shared_epoch_end(outputs, stage=Stage.validate)

    def shared_epoch_end(
        self, output_results: list[SharedStepOut | CfSharedStepOut], *, stage: Stage
    ) -> None:
        self.all_x = torch.cat([_r.x for _r in output_results], 0)
        all_z = torch.cat([_r.z for _r in output_results], 0)
        self.all_s = torch.cat([_r.s for _r in output_results], 0)
        self.all_recon = torch.cat([_r.recon for _r in output_results], 0)
        self.all_cf_pred = torch.cat([_r.cf_pred for _r in output_results], 0)

        if self.debug:
            make_plot(
                x=all_z.clone(),
                s=self.all_s.clone(),
                logger=self.logger,
                name=f"{stage}_z",
                cols=[str(i) for i in range(self.latent_dims)],
            )

        if isinstance(output_results[0], CfSharedStepOut):
            all_cf_x = torch.cat([_r.cf_x for _r in output_results], 0)  # type: ignore[union-attr]
            cf_recon = torch.cat([_r.cf_recon for _r in output_results], 0)  # type: ignore[union-attr]
            if self.debug:
                make_plot(
                    x=all_cf_x.clone(),
                    s=self.all_s.clone(),
                    logger=self.logger,
                    name=f"{stage}_true_counterfactual",
                    cols=self.data_cols,
                )
                make_plot(
                    x=cf_recon.clone(),
                    s=self.all_s.clone(),
                    logger=self.logger,
                    name=f"{stage}_cf_recons",
                    cols=self.data_cols,
                )

    @implements(pl.LightningModule)
    def configure_optimizers(
        self,
    ) -> tuple[list[optim.Optimizer], list[optim.lr_scheduler.ExponentialLR]]:
        opt = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        sched = optim.lr_scheduler.ExponentialLR(opt, gamma=0.999)
        return [opt], [sched]

    @implements(CommonModel)
    def get_recon(self, dataloader: DataLoader) -> np.ndarray:
        recons: list[Tensor] | None = None
        for batch in dataloader:
            x = batch.x.to(self.device)
            s = batch.s.to(self.device)
            enc_fwd = self.forward(x=x, s=s)
            recon = self.invert(index_by_s(enc_fwd.x, s), x)
            if recons is None:
                recons = [recon]
            else:
                recons.append(recon)
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
            enc_fwd = self.forward(x=x, s=s)
            recon_0 = self.invert(enc_fwd.x[0], x)
            recon_1 = self.invert(enc_fwd.x[1], x)
            if recons is None:
                recons = [torch.stack([recon_0, recon_1])]
            else:
                recons.append(torch.stack([recon_0, recon_1]))
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

    def mmd_reporting(
        self, enc_fwd: EncFwd, *, batch: Batch | CfBatch | TernarySample
    ) -> MmdReportingResults:
        with torch.no_grad():
            recon_mmd = mmd2(
                batch.x,
                self.invert(index_by_s(enc_fwd.x, batch.s), batch.x),
                kernel=self.mmd_kernel,
            )

            cf_mmd = mmd2(
                batch.x,
                self.invert(index_by_s(enc_fwd.x, 1 - batch.s), batch.x),
                kernel=self.mmd_kernel,
            )

            s0_dist_mmd = mmd2(
                batch.x[batch.s == 0],
                self.invert(index_by_s(enc_fwd.x, torch.zeros_like(batch.s)), batch.x),
                kernel=self.mmd_kernel,
                biased=True,
            )
            s1_dist_mmd = mmd2(
                batch.x[batch.s == 1],
                self.invert(index_by_s(enc_fwd.x, torch.ones_like(batch.s)), batch.x),
                kernel=self.mmd_kernel,
                biased=True,
            )
        return MmdReportingResults(
            recon=recon_mmd, cf_recon=cf_mmd, s0_dist=s0_dist_mmd, s1_dist=s1_dist_mmd
        )
