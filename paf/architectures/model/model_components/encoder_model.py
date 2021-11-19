"""Encoder model."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import logging
from typing import Any, NamedTuple

from conduit.data import TernarySample
from conduit.fair.data import EthicMlDataModule
from conduit.types import Stage
import numpy as np
import pytorch_lightning as pl
from ranzen import implements, parsable
from sklearn.preprocessing import MinMaxScaler
import torch
from torch import Tensor, nn, no_grad, optim
from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy, l1_loss
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torchmetrics import MeanSquaredError

from paf.base_templates import BaseDataModule
from paf.base_templates.dataset_utils import Batch, CfBatch
from paf.mmd import KernelType, mmd2
from paf.plotting import make_plot
from paf.utils import HistoryPool, Stratifier

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
    # cyc_x: list[Tensor]
    # cyc_z: Tensor


class Loss:
    def __init__(
        self,
        *,
        feature_groups: dict[str, list[slice]] | None = None,
        adv_weight: float = 1.0,
        mmd_weight: float = 1.0,
        cycle_weight: float = 1.0,
        recon_weight: float = 1.0,
        proxy_weight: float = 1.0,
    ):
        self._recon_loss_fn = nn.L1Loss(reduction="mean")
        self.feature_groups = feature_groups if feature_groups is not None else {}
        self._adv_weight = adv_weight
        self._mmd_weight = mmd_weight
        self._cycle_weight = cycle_weight
        self._recon_weight = recon_weight
        self._proxy_weight = proxy_weight
        self._cycle_loss_fn = nn.L1Loss(reduction="mean")
        self._proxy_loss_fn = nn.L1Loss(reduction="none")

    def recon_loss(self, recons: list[Tensor], *, x: Tensor, s: Tensor) -> Tensor:
        if self.feature_groups["discrete"]:
            recon_loss = x.new_tensor(0.0)
            for i in range(
                x[:, slice(self.feature_groups["discrete"][-1].stop, x.shape[1])].shape[1]
            ):
                recon_loss += self._recon_loss_fn(
                    index_by_s(recons, s)[
                        :, slice(self.feature_groups["discrete"][-1].stop, x.shape[1])
                    ][:, i].sigmoid(),
                    x[:, slice(self.feature_groups["discrete"][-1].stop, x.shape[1])][:, i],
                )
            for group_slice in self.feature_groups["discrete"]:
                recon_loss += cross_entropy(
                    index_by_s(recons, s)[:, group_slice],
                    torch.argmax(x[:, group_slice], dim=-1),
                    reduction="mean",
                )
        else:
            recon_loss = self._recon_loss_fn(index_by_s(recons, s).sigmoid(), x)

        return recon_loss * self._recon_weight

    def proxy_loss(
        self, enc_fwd: EncFwd, *, batch: Batch | CfBatch | TernarySample, mask: Tensor | None
    ) -> Tensor:
        _ = (batch,)
        proxy_loss = (
            (mask * self._proxy_loss_fn(enc_fwd.x[0], enc_fwd.x[1])).mean()
            if mask is not None
            else torch.tensor(0.0)
        )
        return self._proxy_weight * proxy_loss

    def adv_loss(self, enc_fwd: EncFwd, *, s: Tensor) -> Tensor:
        return binary_cross_entropy_with_logits(
            enc_fwd.s.squeeze(-1), s, reduction="mean"
        )  # * self._adv_weight

    def mmd_loss(self, enc_fwd: EncFwd, *, s: Tensor, kernel: KernelType) -> Tensor:
        if self._mmd_weight == 0.0:
            return torch.tensor(0.0)

        return mmd2(enc_fwd.z[s == 0], enc_fwd.z[s == 1], kernel=kernel) * self._mmd_weight

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
    decoders: Decoder  # nn.ModuleList
    loss: Loss
    indices: Tensor
    feature_groups: dict[str, list[slice]]
    data_dim: int

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
        mmd_weight: float,
        cycle_weight: float,
        target_weight: float,
        proxy_weight: float,
        lr: float,
        mmd_kernel: KernelType,
        scheduler_rate: float,
        weight_decay: float,
        debug: bool,
        batch_size: int,
    ):
        super().__init__(name="Enc")

        self._adv_weight = adv_weight
        self._mmd_weight = mmd_weight
        self._cycle_weight = cycle_weight
        self._recon_weight = target_weight
        self._proxy_weight = proxy_weight
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

        self.fit_mse = MeanSquaredError()
        self.val_mse = MeanSquaredError()
        self.test_mse = MeanSquaredError()

        self.pool_x0 = Stratifier(pool_size=batch_size // 2)
        self.pool_x1 = Stratifier(pool_size=batch_size // 2)

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
        data: BaseDataModule | EthicMlDataModule,
        indices: list[int],
    ) -> None:
        _ = (cf_available, data)
        self.data_cols = outcome_cols
        self.indices = indices
        self.feature_groups = feature_groups
        self.data_dim = data_dim

        self.adv = Adversary(
            latent_dim=self.latent_dims,
            out_size=1,
            blocks=self.adv_blocks,
            hid_multiplier=self.latent_multiplier,
            weight=self._adv_weight,
        )

        self.in_adv0 = Adversary(
            latent_dim=self.data_dim,
            out_size=1,
            blocks=self.adv_blocks,
            hid_multiplier=self.latent_multiplier,
            weight=0.1,
        )
        self.in_adv1 = Adversary(
            latent_dim=self.data_dim,
            out_size=1,
            blocks=self.adv_blocks,
            hid_multiplier=self.latent_multiplier,
            weight=0.1,
        )
        self.enc = Encoder(
            in_size=self.data_dim + s_dim if self.s_as_input else self.data_dim,
            latent_dim=self.latent_dims,
            blocks=self.encoder_blocks,
            hid_multiplier=self.latent_multiplier,
        )
        self.decoders = nn.ModuleList(
            [
                Decoder(
                    latent_dim=self.latent_dims,
                    in_size=self.data_dim,
                    blocks=self.decoder_blocks,
                    hid_multiplier=self.latent_multiplier,
                )
                for _ in range(num_s)
            ]
        )
        # self.decoders = Decoder(
        #     latent_dim=self.latent_dims + s_dim,
        #     in_size=self.data_dim,
        #     blocks=self.decoder_blocks,
        #     hid_multiplier=self.latent_multiplier,
        # )
        self.loss = Loss(
            feature_groups=self.feature_groups,
            adv_weight=self._adv_weight,
            mmd_weight=self._mmd_weight,
            cycle_weight=self._cycle_weight,
            recon_weight=self._recon_weight,
            proxy_weight=self._proxy_weight,
        )
        self.built = True

    @implements(nn.Module)
    def forward(self, x: Tensor, *, s: Tensor, constraint_mask: Tensor | None = None) -> EncFwd:
        assert self.built
        _x = torch.cat([x, s[..., None]], dim=1) if self.s_as_input else x
        z = self.enc.forward(_x)

        s_pred = self.adv.forward(z)
        # _mask = torch.zeros_like(x) if constraint_mask is None else constraint_mask
        recons = [
            dec(z)  # self.decoders(torch.cat([z, torch.ones_like(s[..., None]) * i], dim=1))
            for i, dec in enumerate(self.decoders)
        ]

        # cycle_x = (
        #     torch.cat([index_by_s(recons, 1 - s), 1 - s[..., None]], dim=1)
        #     if self.s_as_input
        #     else index_by_s(recons, 1 - s)
        # )
        # cycle_z = self.enc.forward(cycle_x)
        # cycle_dec = [dec(cycle_z) for dec in self.decoders]
        return EncFwd(z=z, s=s_pred, x=recons)  # , cyc_z=cycle_z, cyc_x=cycle_dec)

    def make_mask(self, x: Tensor) -> Tensor:

        mask = []
        if self.feature_groups["discrete"]:
            for group in self.feature_groups["discrete"]:
                prob = torch.bernoulli(torch.rand((x.shape[0], 1), device=self.device))
                probs = torch.cat([prob for _ in range(group.start, group.stop)], dim=1)
                mask.append(probs)
            mask.append(
                torch.bernoulli(
                    torch.rand(
                        (
                            x.shape[0],
                            self.data_dim - self.feature_groups["discrete"][-1].stop,
                        ),
                        device=self.device,
                    )
                )
            )
        else:
            mask.append(torch.bernoulli(torch.rand_like(x)))
        return torch.cat(mask, dim=1)

    @implements(pl.LightningModule)
    def training_step(self, batch: Batch | CfBatch | TernarySample, *_: Any) -> Tensor:
        assert self.built

        x0 = self.pool_x0.push_and_pop(batch.x[batch.s == 0])
        s0 = batch.x.new_zeros((x0.shape[0]))
        x1 = self.pool_x1.push_and_pop(batch.x[batch.s == 1])
        s1 = batch.x.new_ones((x1.shape[0]))
        x = torch.cat([x0, x1], dim=0)
        s = torch.cat([s0, s1], dim=0)
        # x = batch.x
        # s = batch.s

        # constraint_mask = torch.ones_like(batch.x) * torch.bernoulli(torch.rand_like(batch.x[0]))
        constraint_mask = self.make_mask(x) if self._proxy_weight > 0.0 else None
        # constraint_mask = torch.zeros_like(batch.x)
        # constraint_mask[:, self.indices] += 1
        enc_fwd = self.forward(x=x, s=s, constraint_mask=constraint_mask)

        recon_loss = self.loss.recon_loss(recons=enc_fwd.x, x=x, s=s)
        # proxy_loss = self.loss.proxy_loss(enc_fwd, batch=batch, mask=constraint_mask)
        adv_loss = self.loss.adv_loss(enc_fwd=enc_fwd, s=s)
        mmd_loss = self.loss.mmd_loss(enc_fwd=enc_fwd, s=s, kernel=self.mmd_kernel)
        # report_of_cyc_loss, cycle_loss = self.loss.cycle_loss(cyc_x=enc_fwd.cyc_x, batch=batch)

        loss = recon_loss + adv_loss + mmd_loss  # + proxy_loss  # + cycle_loss
        x0_adv = torch.nn.functional.binary_cross_entropy_with_logits(
            torch.cat(
                [
                    self.in_adv0(enc_fwd.x[0][s == 0].detach()).squeeze(-1),
                    self.in_adv0(enc_fwd.x[0][s == 1]).squeeze(-1),
                ],
                dim=0,
            ),
            torch.cat([s[s == 0], s[s == 1]], dim=0),
        )
        x1_adv = torch.nn.functional.binary_cross_entropy_with_logits(
            torch.cat(
                [
                    self.in_adv1(enc_fwd.x[1][s == 0]).squeeze(-1),
                    self.in_adv1(enc_fwd.x[1][s == 1].detach()).squeeze(-1),
                ],
                dim=0,
            ),
            torch.cat([s[s == 0], s[s == 1]], dim=0),
        )
        loss += x0_adv + x1_adv
        # mmd_results = self.mmd_reporting(enc_fwd=enc_fwd, batch=batch)

        to_log = {
            f"{Stage.fit}/enc/loss": loss,
            f"{Stage.fit}/enc/recon_loss": recon_loss,
            f"{Stage.fit}/enc/mmd_loss": mmd_loss,
            f"{Stage.fit}/enc/adv_loss": adv_loss,
            f"{Stage.fit}/enc/x0_adv_loss": x0_adv,
            f"{Stage.fit}/enc/x1_adv_loss": x1_adv,
            # f"{Stage.fit}/enc/proxy_loss": proxy_loss,
            f"{Stage.fit}/enc/mse": self.fit_mse(self.invert(index_by_s(enc_fwd.x, s), x), x),
            f"{Stage.fit}/enc/z_norm": enc_fwd.z.detach().norm(dim=1).mean(),
            f"{Stage.fit}/enc/z_mean_abs_diff": (
                enc_fwd.z[s <= 0].detach().mean() - enc_fwd.z[s > 0].detach().mean()
            ).abs(),
            # f"{Stage.fit}/enc/cycle_loss": report_of_cyc_loss,
            # f"{Stage.fit}/enc/recon_mmd": mmd_results.recon,
            # f"{Stage.fit}/enc/cf_recon_mmd": mmd_results.cf_recon,
            # f"{Stage.fit}/enc/s0_dist_mmd": mmd_results.s0_dist,
            # f"{Stage.fit}/enc/s1_dist_mmd": mmd_results.s1_dist,
        }

        # if isinstance(batch, CfBatch):
        #     with no_grad():
        # enc_fwd = self.forward(x=batch.cfx, s=batch.cfs)
        # cf_recon_loss = l1_loss(
        #     index_by_s(enc_fwd.x, batch.cfs).sigmoid(), batch.cfx, reduction="mean"
        # )
        # to_log[f"{Stage.fit}/enc/cf_recon_loss"] = cf_recon_loss

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
        constraint_mask = torch.zeros_like(batch.x)
        constraint_mask[:, self.indices] += 1
        enc_fwd = self.forward(x=batch.x, s=batch.s, constraint_mask=constraint_mask)

        recon_loss = self.loss.recon_loss(recons=enc_fwd.x, x=batch.x, s=batch.s)
        adv_loss = self.loss.adv_loss(enc_fwd=enc_fwd, s=batch.s)
        mmd_loss = self.loss.mmd_loss(enc_fwd=enc_fwd, s=batch.s, kernel=self.mmd_kernel)
        # cycle_loss, _ = self.loss.cycle_loss(cyc_x=enc_fwd.cyc_x, batch=batch)

        loss = recon_loss + adv_loss + mmd_loss  # + cycle_loss

        # mmd_results = self.mmd_reporting(enc_fwd=enc_fwd, batch=batch)

        mse = self.val_mse if stage is Stage.validate else self.test_mse

        self.log(f"{stage}/enc/loss", loss)
        self.log(f"{stage}/enc/recon_loss", recon_loss)
        # self.log(f"{stage}/enc/cycle_loss", cycle_loss)
        self.log(f"{stage}/enc/mmd_loss", mmd_loss)
        self.log(
            f"{stage}/enc/mse", mse(self.invert(index_by_s(enc_fwd.x, batch.s), batch.x), batch.x)
        )
        # self.log(f"{stage}/enc/recon_mmd", mmd_results.recon)
        # self.log(f"{stage}/enc/cf_recon_mmd", mmd_results.cf_recon)
        # self.log(f"{stage}/enc/s0_dist_mmd", mmd_results.s0_dist)
        # self.log(f"{stage}/enc/s1_dist_mmd", mmd_results.s1_dist)

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
                x=self.all_recon.clone(),
                s=self.all_s.clone(),
                logger=self.logger,
                name=f"{stage}_recon",
                cols=self.data_cols,
            )
            make_plot(
                x=self.all_x.clone(),
                s=self.all_s.clone(),
                logger=self.logger,
                name=f"{stage}_true_x",
                cols=self.data_cols,
            )
            make_plot(
                x=all_z.clone(),
                s=self.all_s.clone(),
                logger=self.logger,
                name=f"{stage}_z",
                cols=[str(i) for i in range(self.latent_dims)],
            )

        if isinstance(output_results[0], CfSharedStepOut):
            self.all_cf_x = torch.cat([_r.cf_x for _r in output_results], 0)  # type: ignore[union-attr]
            self.cf_recon = torch.cat([_r.cf_recon for _r in output_results], 0)  # type: ignore[union-attr]
            if self.debug:
                make_plot(
                    x=self.all_cf_x.clone(),
                    s=self.all_s.clone(),
                    logger=self.logger,
                    name=f"{stage}_true_counterfactual",
                    cols=self.data_cols,
                )
                make_plot(
                    x=self.cf_recon.clone(),
                    s=self.all_s.clone(),
                    logger=self.logger,
                    name=f"{stage}_cf_recons",
                    cols=self.data_cols,
                )

    @implements(pl.LightningModule)
    def configure_optimizers(
        self,
    ) -> tuple[list[optim.Optimizer], list[ExponentialLR]]:
        opt = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        sched = ExponentialLR(opt, gamma=self.scheduler_rate)
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

    # def mmd_reporting(
    #     self, enc_fwd: EncFwd, *, batch: Batch | CfBatch | TernarySample
    # ) -> MmdReportingResults:
    #     with torch.no_grad():
    #         recon_mmd = mmd2(
    #             batch.x,
    #             self.invert(index_by_s(enc_fwd.x, batch.s), batch.x),
    #             kernel=self.mmd_kernel,
    #         )
    #
    #         if isinstance(batch, CfBatch):
    #             cf_mmd = mmd2(
    #                 batch.x,
    #                 self.invert(index_by_s(enc_fwd.x, 1 - batch.s), batch.cfx),
    #                 kernel=self.mmd_kernel,
    #             )
    #         else:
    #             cf_mmd = torch.tensor(-10)
    #
    #         s0_dist_mmd = mmd2(
    #             batch.x[batch.s == 0],
    #             self.invert(enc_fwd.x[0], batch.x),
    #             kernel=self.mmd_kernel,
    #             biased=True,
    #         )
    #         s1_dist_mmd = mmd2(
    #             batch.x[batch.s == 1],
    #             self.invert(enc_fwd.x[1], batch.x),
    #             kernel=self.mmd_kernel,
    #             biased=True,
    #         )
    #     return MmdReportingResults(
    #         recon=recon_mmd, cf_recon=cf_mmd, s0_dist=s0_dist_mmd, s1_dist=s1_dist_mmd
    #     )
