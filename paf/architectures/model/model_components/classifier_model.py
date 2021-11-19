"""Encoder model."""
from __future__ import annotations
from typing import Any, NamedTuple, Union

from conduit.data import TernarySample
from conduit.types import Stage
import numpy as np
import pytorch_lightning as pl
from ranzen import implements, parsable, str_to_enum
from ranzen.torch.transforms import RandomMixUp
from sklearn.preprocessing import MinMaxScaler
import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

__all__ = ["BaseModel", "Adversary", "Clf", "ClfInferenceOut", "ClfFwd"]

from torchmetrics import Accuracy

from paf.base_templates import Batch, CfBatch
from paf.mmd import KernelType, mmd2
from paf.plotting import make_plot
from paf.utils import HistoryPool, Stratifier

from .common_model import Adversary, BaseModel, CommonModel, Decoder, Encoder
from .model_utils import index_by_s


class ClfFwd(NamedTuple):
    z: Tensor
    s: Tensor
    y: list[Tensor]


class ClfInferenceOut(NamedTuple):
    y: Tensor
    z: Tensor
    s: Tensor
    preds: Tensor
    preds_0: Tensor
    preds_1: Tensor
    cf_y: Tensor | None
    cf_preds: Tensor | None


class Loss:
    def __init__(
        self,
        adv_weight: float = 1.0,
        pred_weight: float = 1.0,
        mmd_weight: float = 1.0,
        kernel: KernelType = KernelType.LINEAR,
    ):
        self._adv_weight = adv_weight
        self._pred_weight = pred_weight
        self._mmd_weight = mmd_weight
        self._kernel = kernel

        self._pred_loss_fn = nn.BCEWithLogitsLoss
        self._adv_loss_fn = nn.BCEWithLogitsLoss

    def pred_loss(self, clf_fwd: ClfFwd, s: Tensor, y: Tensor, weight: Tensor | None) -> Tensor:
        return self._pred_weight * self._pred_loss_fn(reduction="mean", weight=weight)(
            index_by_s(clf_fwd.y, s).squeeze(-1), y
        )

    def mmd_loss(self, clf_fwd: ClfFwd, s: Tensor) -> Tensor:
        if self._mmd_weight == 0.0:
            return torch.tensor(0.0)
        return self._mmd_weight * mmd2(clf_fwd.z[s == 0], clf_fwd.z[s == 1], kernel=self._kernel)

    def adv_loss(self, clf_fwd: ClfFwd, s: Tensor) -> Tensor:
        return self._adv_loss_fn(reduction="mean")(clf_fwd.s.squeeze(-1), s)  # * self._adv_weight


class Clf(CommonModel):
    """Main Autoencoder."""

    cf_model: bool
    outcome_cols: list[str]
    enc: nn.Module
    decoders: nn.ModuleList
    built: bool
    adv: Adversary

    @parsable
    def __init__(
        self,
        adv_weight: float,
        pred_weight: float,
        mmd_weight: float,
        lr: float,
        s_as_input: bool,
        latent_dims: int,
        mmd_kernel: Union[str, KernelType],
        scheduler_rate: float,
        weight_decay: float,
        use_iw: bool,
        encoder_blocks: int,
        adv_blocks: int,
        decoder_blocks: int,
        latent_multiplier: int,
        batch_size: int,
        debug: bool,
    ):
        """Classifier."""
        super().__init__(name="Clf")

        self.learning_rate = lr
        self.s_as_input = s_as_input
        self.latent_dims = latent_dims
        self.scheduler_rate = scheduler_rate
        self.weight_decay = weight_decay
        self.use_iw = use_iw
        self.encoder_blocks = encoder_blocks
        self.adv_blocks = adv_blocks
        self.decoder_blocks = decoder_blocks
        self.latent_multiplier = latent_multiplier
        self.debug = debug

        self.fit_acc = Accuracy()
        self.fit_cf_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

        self._adv_weight = adv_weight

        self.loss = Loss(
            adv_weight=adv_weight,
            pred_weight=pred_weight,
            mmd_weight=mmd_weight,
            kernel=str_to_enum(mmd_kernel, enum=KernelType),
        )

        self.mixup = RandomMixUp(
            lambda_sampler=torch.distributions.Uniform(0.0, 0.49), num_classes=2
        )
        # self.mixup_s0 = RandomMixUp(
        #     lambda_sampler=torch.distributions.Uniform(0.0, 1.0), num_classes=2
        # )
        # self.mixup_s1 = RandomMixUp(
        #     lambda_sampler=torch.distributions.Uniform(0.0, 1.0), num_classes=2
        # )

        self.pool_x_s0y0 = Stratifier(pool_size=batch_size // 4)
        self.pool_x_s0y1 = Stratifier(pool_size=batch_size // 4)
        self.pool_x_s1y0 = Stratifier(pool_size=batch_size // 4)
        self.pool_x_s1y1 = Stratifier(pool_size=batch_size // 4)

        self.built = False

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
        scaler: MinMaxScaler | None,
    ) -> None:
        _ = (feature_groups, scaler)
        self.cf_model = cf_available
        self.outcome_cols = outcome_cols
        self.enc = Encoder(
            in_size=(data_dim + s_dim) if self.s_as_input else data_dim,
            latent_dim=self.latent_dims,
            blocks=self.encoder_blocks,
            hid_multiplier=self.latent_multiplier,
        )
        self.adv = Adversary(
            latent_dim=self.latent_dims,
            out_size=1,
            blocks=self.adv_blocks,
            hid_multiplier=self.latent_multiplier,
            weight=self._adv_weight,
        )
        self.in_adv0 = Adversary(
            latent_dim=1,
            out_size=1,
            blocks=self.adv_blocks,
            hid_multiplier=self.latent_multiplier,
            weight=0.1,
        )
        self.in_adv1 = Adversary(
            latent_dim=1,
            out_size=1,
            blocks=self.adv_blocks,
            hid_multiplier=self.latent_multiplier,
            weight=0.1,
        )
        self.decoders = nn.ModuleList(
            [
                Decoder(
                    latent_dim=self.latent_dims,
                    in_size=1,
                    blocks=self.decoder_blocks,
                    hid_multiplier=self.latent_multiplier,
                )
                for _ in range(num_s)
            ]
        )
        # self.decoders = Decoder(
        #     latent_dim=self.latent_dims + s_dim,
        #     in_size=1,
        #     blocks=self.decoder_blocks,
        #     hid_multiplier=self.latent_multiplier,
        # )
        self.built = True

    @implements(nn.Module)
    def forward(self, x: Tensor, *, s: Tensor) -> ClfFwd:
        assert self.built
        _x = torch.cat([x, s[..., None]], dim=1) if self.s_as_input else x
        z = self.enc.forward(_x)
        s_pred = self.adv.forward(z)
        preds = [dec(z) for dec in self.decoders]
        # preds = [
        #     self.decoders(torch.cat([z, torch.ones_like(s[..., None]) * i], dim=1))
        #     for i in range(2)
        # ]
        return ClfFwd(z=z, s=s_pred, y=preds)

    @implements(pl.LightningModule)
    def training_step(self, batch: Batch | CfBatch | TernarySample, *_: Any) -> Tensor:
        assert self.built

        x_s0y0 = self.pool_x_s0y0.push_and_pop(batch.x[(batch.s == 0) & (batch.y == 0)])
        x_s0y1 = self.pool_x_s0y1.push_and_pop(batch.x[(batch.s == 0) & (batch.y == 1)])
        assert len(x_s0y0) == len(x_s0y1)

        s_s0y0 = batch.x.new_zeros((x_s0y0.shape[0]))
        s_s0y1 = batch.x.new_zeros((x_s0y1.shape[0]))
        assert len(s_s0y0) == len(s_s0y1)

        y_s0y0 = batch.x.new_zeros((x_s0y0.shape[0]))
        y_s0y1 = batch.x.new_ones((x_s0y1.shape[0]))
        x_s1y0 = self.pool_x_s1y0.push_and_pop(batch.x[(batch.s == 1) & (batch.y == 0)])
        x_s1y1 = self.pool_x_s1y1.push_and_pop(batch.x[(batch.s == 1) & (batch.y == 1)])
        s_s1y0 = batch.x.new_ones((x_s1y0.shape[0]))
        s_s1y1 = batch.x.new_ones((x_s1y1.shape[0]))
        y_s1y0 = batch.x.new_zeros((x_s1y0.shape[0]))
        y_s1y1 = batch.x.new_ones((x_s1y1.shape[0]))

        x_s0 = torch.cat([x_s0y0, x_s0y1], dim=0)
        s_s0 = torch.cat([s_s0y0, s_s0y1], dim=0)
        y_s0 = torch.cat([y_s0y0, y_s0y1], dim=0)
        # mixed_s0 = self.mixup_s0(x_s0, targets=y_s0.long())

        x_s1 = torch.cat([x_s1y0, x_s1y1], dim=0)
        s_s1 = torch.cat([s_s1y0, s_s1y1], dim=0)
        y_s1 = torch.cat([y_s1y0, y_s1y1], dim=0)
        # mixed_s1 = self.mixup_s1(x_s1, targets=y_s1.long())

        # mixed_x = torch.cat([mixed_s0.inputs, mixed_s1.inputs], dim=0)
        s = torch.cat([s_s0, s_s1], dim=0)
        # mixed_y = torch.cat([mixed_s0.targets[:, 1], mixed_s1.targets[:, 1]], dim=0)

        x = torch.cat([x_s0, x_s1], dim=0)
        y = torch.cat([y_s0, y_s1], dim=0)

        # mixed_out = self.forward(x=batch.x, s=batch.s)
        # mixed_pred_loss = torch.nn.functional.mse_loss(
        #     index_by_s(mixed_out.y, s).squeeze().sigmoid(), mixed_y
        # )

        # x = batch.x
        # s = batch.s
        # y = batch.y

        clf_out = self.forward(x=x, s=s)
        _iw = batch.iw if self.use_iw and isinstance(batch, (Batch, CfBatch)) else None
        pred_loss = self.loss.pred_loss(clf_out, s=s, y=y, weight=_iw)
        adv_loss = self.loss.adv_loss(clf_out, s=s)
        mmd_loss = self.loss.mmd_loss(clf_out, s=s)

        # x_s0y0 = batch.x[(batch.s == 0) & (batch.y == 0)]
        # x_s0y1 = batch.x[(batch.s == 0) & (batch.y == 1)]
        # x_s0 = torch.cat([x_s0y0, x_s0y1], dim=0)
        # x_s1y0 = batch.x[(batch.s == 1) & (batch.y == 0)]
        # x_s1y1 = batch.x[(batch.s == 1) & (batch.y == 1)]
        # x_s1 = torch.cat([x_s1y0, x_s1y1], dim=0)
        # s_s0y0 = batch.x.new_zeros((x_s0y0.shape[0]))
        # s_s0y1 = batch.x.new_zeros((x_s0y1.shape[0]))
        # s_s0 = torch.cat([s_s0y0, s_s0y1], dim=0)
        # s_s1y0 = batch.x.new_ones((x_s1y0.shape[0]))
        # s_s1y1 = batch.x.new_ones((x_s1y1.shape[0]))
        # s_s1 = torch.cat([s_s1y0, s_s1y1], dim=0)
        # y_s0y0 = batch.x.new_zeros((x_s0y0.shape[0]))
        # y_s0y1 = batch.x.new_ones((x_s0y1.shape[0]))
        # y_s0 = torch.cat([y_s0y0, y_s0y1], dim=0)
        # y_s1y0 = batch.x.new_zeros((x_s1y0.shape[0]))
        # y_s1y1 = batch.x.new_ones((x_s1y1.shape[0]))
        # y_s1 = torch.cat([y_s1y0, y_s1y1], dim=0)

        # mixed_s0 = self.mixup_s0(x_s0, targets=y_s0.long())
        # mixed_s1 = self.mixup_s1(x_s1, targets=y_s1.long())
        # mixed_x = torch.cat([mixed_s0.inputs, mixed_s1.inputs], dim=0)
        # mixed_s = torch.cat([s_s0, s_s1], dim=0)
        # mixed_y = torch.cat([mixed_s0.targets[:, 1], mixed_s1.targets[:, 1]], dim=0)

        mixed = self.mixup(x, targets=y.long(), group_labels=s.long())
        mixed_out = self.forward(x=mixed.inputs, s=s)
        mixed_pred_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            index_by_s(mixed_out.y, s).squeeze(), mixed.targets[:, 1]
        )

        loss = mixed_pred_loss + adv_loss + mmd_loss + pred_loss

        # x0_adv = torch.nn.functional.binary_cross_entropy_with_logits(
        #     torch.cat(
        #         [
        #             self.in_adv0(clf_out.y[0][s == 0].detach()).squeeze(-1),
        #             self.in_adv0(clf_out.y[0][s == 1]).squeeze(-1),
        #         ],
        #         dim=0,
        #     ),
        #     torch.cat([s[s == 0], s[s == 1]], dim=0),
        # )
        # x0_adv += mmd2(
        #     clf_out.y[0][s == 0].detach(),
        #     clf_out.y[0][s == 1],
        #     kernel=KernelType.RBF,
        # )
        #
        # x1_adv = torch.nn.functional.binary_cross_entropy_with_logits(
        #     torch.cat(
        #         [
        #             self.in_adv1(clf_out.y[1][s == 0]).squeeze(-1),
        #             self.in_adv1(clf_out.y[1][s == 1].detach()).squeeze(-1),
        #         ],
        #         dim=0,
        #     ),
        #     torch.cat([s[s == 0], s[s == 1]], dim=0),
        # )
        # x1_adv += mmd2(
        #     clf_out.y[1][s == 0],
        #     clf_out.y[1][s == 1].detach(),
        #     kernel=KernelType.RBF,
        # )

        # loss += x0_adv + x1_adv

        to_log = {
            f"{Stage.fit}/clf/acc": self.fit_acc(
                index_by_s(clf_out.y, s).squeeze(-1).sigmoid(), y.int()
            ),
            f"{Stage.fit}/clf/loss": loss,
            f"{Stage.fit}/clf/pred_loss": pred_loss,
            f"{Stage.fit}/clf/adv_loss": adv_loss,
            # f"{Stage.fit}/clf/y0_adv_loss": x0_adv,
            # f"{Stage.fit}/clf/y1_adv_loss": x1_adv,
            f"{Stage.fit}/clf/mmd_loss": mmd_loss,
            f"{Stage.fit}/clf/z_norm": clf_out.z.detach().norm(dim=1).mean(),
            f"{Stage.fit}/clf/z_mean_abs_diff": (
                clf_out.z[s <= 0].detach().mean() - clf_out.z[s > 0].detach().mean()
            ).abs(),
        }

        # if isinstance(batch, CfBatch):
        #     with torch.no_grad():
        #         to_log[f"{Stage.fit}/clf/cf_acc"] = self.fit_cf_acc(
        #             index_by_s(clf_out.y, batch.cfs).squeeze(-1).sigmoid(), batch.cfy.int()
        #         )
        # cf_recon_loss = l1_loss(
        #     index_by_s(enc_fwd.x, batch.cfs).sigmoid(), batch.cfx, reduction="mean"
        # )
        # to_log[f"{Stage.fit}/enc/cf_recon_loss"] = cf_recon_loss

        self.log_dict(to_log, logger=True)

        return loss

    @staticmethod
    def threshold(z: Tensor) -> Tensor:
        """Go from soft to discrete features."""
        with torch.no_grad():
            return z.sigmoid().round()

    @implements(pl.LightningModule)
    def validation_step(self, batch: Batch | CfBatch | TernarySample, *_: Any) -> ClfInferenceOut:
        return self.shared_step(batch, stage=Stage.validate)

    @implements(pl.LightningModule)
    def test_step(self, batch: Batch | CfBatch | TernarySample, *_: Any) -> ClfInferenceOut:
        return self.shared_step(batch, stage=Stage.test)

    def shared_step(
        self, batch: Batch | CfBatch | TernarySample, *, stage: Stage
    ) -> ClfInferenceOut:
        assert self.built
        clf_out = self.forward(x=batch.x, s=batch.s)
        acc = self.val_acc if stage is Stage.validate else self.test_acc
        self.log(
            f"{stage}/clf/acc",
            acc(self.threshold(index_by_s(clf_out.y, batch.s).squeeze(-1)), batch.y.int()),
        )

        return ClfInferenceOut(
            y=batch.y,
            z=clf_out.z,
            s=batch.s,
            preds=index_by_s(clf_out.y, batch.s).sigmoid(),
            preds_0=clf_out.y[0].sigmoid(),
            preds_1=clf_out.y[1].sigmoid(),
            cf_y=batch.cfy if isinstance(batch, CfBatch) else None,
            cf_preds=index_by_s(clf_out.y, 1 - batch.s).sigmoid()
            if isinstance(batch, CfBatch)
            else None,
        )

    @implements(pl.LightningModule)
    def validation_epoch_end(self, outputs: list[ClfInferenceOut]) -> None:
        return self.shared_epoch_end(outputs, stage=Stage.test)

    @implements(pl.LightningModule)
    def test_epoch_end(self, outputs: list[ClfInferenceOut]) -> None:
        return self.shared_epoch_end(outputs, stage=Stage.test)

    def shared_epoch_end(self, outputs: list[ClfInferenceOut], *, stage: Stage) -> None:
        all_y = torch.cat([_r.y for _r in outputs], 0)
        all_z = torch.cat([_r.z for _r in outputs], 0)
        all_s = torch.cat([_r.s for _r in outputs], 0)
        all_preds = torch.cat([_r.preds for _r in outputs], 0)
        preds_0 = torch.cat([_r.preds_0 for _r in outputs], 0)
        preds_1 = torch.cat([_r.preds_1 for _r in outputs], 0)

        if self.debug:
            make_plot(
                x=all_y.unsqueeze(-1),
                s=all_s,
                logger=self.logger,
                name=f"{stage}/true_data",
                cols=["out"],
            )
            make_plot(
                x=all_preds, s=all_s, logger=self.logger, name=f"{stage}/preds", cols=["preds"]
            )
            make_plot(
                x=preds_0, s=all_s, logger=self.logger, name=f"{stage}/preds_all0", cols=["preds"]
            )
            make_plot(
                x=preds_1, s=all_s, logger=self.logger, name=f"{stage}/preds_all1", cols=["preds"]
            )
            make_plot(
                x=all_z,
                s=all_s,
                logger=self.logger,
                name="z",
                cols=[str(i) for i in range(self.latent_dims)],
            )

        if self.cf_model:
            all_cf_y = torch.cat([_r.cf_y for _r in outputs if _r.cf_y is not None], 0)
            cf_preds = torch.cat([_r.cf_preds for _r in outputs if _r.cf_preds is not None], 0)
            if self.debug:
                make_plot(
                    x=all_cf_y.unsqueeze(-1),
                    s=all_s,
                    logger=self.logger,
                    name=f"{stage}/true_counterfactual_outcome",
                    cols=["preds"],
                )
                make_plot(
                    x=cf_preds,
                    s=all_s,
                    logger=self.logger,
                    name=f"{stage}/cf_preds",
                    cols=["preds"],
                )

    @implements(pl.LightningModule)
    def configure_optimizers(self) -> tuple[list[AdamW], list[ExponentialLR]]:
        opt = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        sched = ExponentialLR(opt, gamma=self.scheduler_rate)
        return [opt], [sched]

    @implements(CommonModel)
    def get_recon(self, dataloader: DataLoader) -> np.ndarray:
        """Should really be called get_preds."""
        preds: list[Tensor] | None = None
        for batch in dataloader:
            x = batch.x.to(self.device)
            s = batch.s.to(self.device)
            clf_out = self.forward(x=x, s=s)
            pred = self.threshold(index_by_s(clf_out.y, s))
            if preds is None:
                preds = [pred]
            else:
                preds.append(pred)
        assert preds is not None
        _preds = torch.cat(preds, dim=0)
        return _preds.detach().cpu().numpy()

    def from_recons(self, recons: list[Tensor]) -> dict[str, tuple[Tensor, ...]]:
        """Given recons, give all possible predictions."""
        preds_dict: dict[str, tuple[Tensor, ...]] = {}

        for i, rec in enumerate(recons):
            z, s_pred, preds = self.forward(x=rec, s=torch.ones_like(rec[:, 0]) * i)
            for _s in range(2):
                preds_dict[f"{i}_{_s}"] = (z, s_pred, self.threshold(preds[_s]))
        return preds_dict
