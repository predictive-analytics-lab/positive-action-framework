"""Encoder model."""
from __future__ import annotations
from typing import Any, NamedTuple, Union

from conduit.data import TernarySample
from conduit.types import Stage
import numpy as np
import pytorch_lightning as pl
from ranzen import implements, parsable, str_to_enum
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

    def pred_loss(
        self, clf_fwd: ClfFwd, batch: Batch | TernarySample, weight: Tensor | None
    ) -> Tensor:
        return self._pred_weight * self._pred_loss_fn(reduction="mean", weight=weight)(
            index_by_s(clf_fwd.y, batch.s).squeeze(-1), batch.y
        )

    def mmd_loss(self, clf_fwd: ClfFwd, batch: Batch | TernarySample) -> Tensor:
        return self._mmd_weight * mmd2(
            clf_fwd.z[batch.s == 0], clf_fwd.z[batch.s == 1], kernel=self._kernel
        )

    def adv_loss(self, clf_fwd: ClfFwd, batch: Batch | TernarySample) -> Tensor:
        return self._adv_loss_fn(reduction="mean")(
            clf_fwd.s.squeeze(-1), batch.s
        )  # * self._adv_weight


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
        # preds = [dec(z) for dec in self.decoders]
        preds = [
            dec(z)  # (torch.cat([z, torch.ones_like(s[..., None]) * i], dim=1))
            for i, dec in enumerate(self.decoders)
        ]
        return ClfFwd(z=z, s=s_pred, y=preds)

    @implements(pl.LightningModule)
    def training_step(self, batch: Batch | CfBatch | TernarySample, *_: Any) -> Tensor:
        assert self.built
        clf_out = self.forward(x=batch.x, s=batch.s)
        _iw = batch.iw if self.use_iw and isinstance(batch, (Batch, CfBatch)) else None
        pred_loss = self.loss.pred_loss(clf_out, batch, weight=_iw)
        adv_loss = self.loss.adv_loss(clf_out, batch)
        mmd_loss = self.loss.mmd_loss(clf_out, batch)
        loss = pred_loss + adv_loss + mmd_loss

        x0_adv = torch.nn.functional.binary_cross_entropy_with_logits(
            torch.cat(
                [
                    self.in_adv0(clf_out.y[0][batch.s == 0].detach()).squeeze(-1),
                    self.in_adv0(clf_out.y[0][batch.s == 1]).squeeze(-1),
                ],
                dim=0,
            ),
            torch.cat([batch.s[batch.s == 0], batch.s[batch.s == 1]], dim=0),
        )
        x1_adv = torch.nn.functional.binary_cross_entropy_with_logits(
            torch.cat(
                [
                    self.in_adv1(clf_out.y[1][batch.s == 0]).squeeze(-1),
                    self.in_adv1(clf_out.y[1][batch.s == 1].detach()).squeeze(-1),
                ],
                dim=0,
            ),
            torch.cat([batch.s[batch.s == 0], batch.s[batch.s == 1]], dim=0),
        )
        loss += x0_adv + x1_adv

        to_log = {
            f"{Stage.fit}/clf/acc": self.fit_acc(
                index_by_s(clf_out.y, batch.s).squeeze(-1).sigmoid(), batch.y.int()
            ),
            f"{Stage.fit}/clf/loss": loss,
            f"{Stage.fit}/clf/pred_loss": pred_loss,
            f"{Stage.fit}/clf/adv_loss": adv_loss,
            f"{Stage.fit}/clf/y0_adv_loss": x0_adv,
            f"{Stage.fit}/clf/y1_adv_loss": x1_adv,
            f"{Stage.fit}/clf/mmd_loss": mmd_loss,
            f"{Stage.fit}/clf/z_norm": clf_out.z.detach().norm(dim=1).mean(),
            f"{Stage.fit}/clf/z_mean_abs_diff": (
                clf_out.z[batch.s <= 0].detach().mean() - clf_out.z[batch.s > 0].detach().mean()
            ).abs(),
        }

        if isinstance(batch, CfBatch):
            with torch.no_grad():
                to_log[f"{Stage.fit}/clf/cf_acc"] = self.fit_cf_acc(
                    index_by_s(clf_out.y, batch.cfs).squeeze(-1).sigmoid(), batch.cfy.int()
                )
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
        assert self.built
        clf_out = self.forward(x=batch.x, s=batch.s)

        self.log(
            f"{Stage.validate}/clf/acc",
            self.val_acc(self.threshold(index_by_s(clf_out.y, batch.s).squeeze(-1)), batch.y.int()),
        )

        return ClfInferenceOut(
            y=batch.y,
            z=clf_out.z,
            s=batch.s,
            preds=self.threshold(index_by_s(clf_out.y, batch.s)),
            preds_0=self.threshold(clf_out.y[0]),
            preds_1=self.threshold(clf_out.y[1]),
            cf_y=batch.cfy if isinstance(batch, CfBatch) else None,
            cf_preds=self.threshold(index_by_s(clf_out.y, batch.cfs))
            if isinstance(batch, CfBatch)
            else None,
        )

    @implements(pl.LightningModule)
    def test_step(self, batch: Batch | CfBatch | TernarySample, *_: Any) -> ClfInferenceOut:
        assert self.built
        clf_out = self.forward(x=batch.x, s=batch.s)
        self.log(
            f"{Stage.test}/clf/acc",
            self.test_acc(
                self.threshold(index_by_s(clf_out.y, batch.s).squeeze(-1)), batch.y.int()
            ),
        )

        return ClfInferenceOut(
            y=batch.y,
            z=clf_out.z,
            s=batch.s,
            preds=self.threshold(index_by_s(clf_out.y, batch.s)),
            preds_0=self.threshold(clf_out.y[0]),
            preds_1=self.threshold(clf_out.y[1]),
            cf_y=batch.cfy if isinstance(batch, CfBatch) else None,
            cf_preds=self.threshold(index_by_s(clf_out.y, batch.cfs))
            if isinstance(batch, CfBatch)
            else None,
        )

    @implements(pl.LightningModule)
    def test_epoch_end(self, outputs: list[ClfInferenceOut]) -> None:
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
                name="true_data",
                cols=["out"],
            )
            make_plot(x=all_preds, s=all_s, logger=self.logger, name="preds", cols=["preds"])
            make_plot(x=preds_0, s=all_s, logger=self.logger, name="preds", cols=["preds"])
            make_plot(x=preds_1, s=all_s, logger=self.logger, name="preds", cols=["preds"])
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
                    name="true_counterfactual_outcome",
                    cols=["preds"],
                )
                make_plot(x=cf_preds, s=all_s, logger=self.logger, name="cf_preds", cols=["preds"])

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
