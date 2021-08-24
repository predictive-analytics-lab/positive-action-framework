"""Encoder model."""
from __future__ import annotations

from typing import NamedTuple

from kit import implements
import numpy as np
from pytorch_lightning import LightningModule
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import Tensor, cat, nn
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.optim import Adam
from torch.utils.data import DataLoader

from paf.base_templates.dataset_utils import Batch, CfBatch
from paf.config_classes.dataclasses import KernelType
from paf.log_progress import do_log
from paf.mmd import mmd2
from paf.model.blocks import block, mid_blocks
from paf.model.common_model import CommonModel
from paf.model.model_utils import grad_reverse, index_by_s
from paf.plotting import make_plot


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


class Clf(CommonModel):
    """Main Autoencoder."""

    cf_model: bool
    outcome_cols: list[str]
    enc: nn.Module
    decoders: nn.ModuleList
    built: bool

    def __init__(
        self,
        adv_weight: float,
        reg_weight: float,
        pred_weight: float,
        lr: float,
        s_as_input: bool,
        latent_dims: int,
        mmd_kernel: KernelType,
        scheduler_rate: float,
        weight_decay: float,
        use_iw: bool,
        encoder_blocks: int,
        adv_blocks: int,
        decoder_blocks: int,
        latent_multiplier: int,
    ):
        super().__init__(name="Clf")

        self.adv_weight = adv_weight
        self.reg_weight = reg_weight
        self.pred_weight = pred_weight
        self.lr = lr
        self.s_as_input = s_as_input
        self.latent_dims = latent_dims
        self.mmd_kernel = mmd_kernel
        self.scheduler_rate = scheduler_rate
        self.weight_decay = weight_decay
        self.use_iw = use_iw
        self.encoder_blocks = encoder_blocks
        self.adv_blocks = adv_blocks
        self.decoder_blocks = decoder_blocks
        self.latent_multiplier = latent_multiplier
        self.built = False

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
        self.built = True

    @implements(nn.Module)
    def forward(self, x: Tensor, s: Tensor) -> ClfFwd:
        assert self.built
        _x = torch.cat([x, s[..., None]], dim=1) if self.s_as_input else x
        z = self.enc(_x)
        s_pred = self.adv(z)
        preds = [dec(z) for dec in self.decoders]
        return ClfFwd(z=z, s=s_pred, y=preds)

    @implements(LightningModule)
    def training_step(self, batch: Batch | CfBatch, batch_idx: int) -> Tensor:
        assert self.built
        clf_out = self.forward(batch.x, batch.s)
        _iw = batch.iw if self.use_iw else None
        pred_loss = binary_cross_entropy_with_logits(
            index_by_s(clf_out.y, batch.s).squeeze(-1), batch.y, reduction="mean", weight=_iw
        )
        adv_loss = (
            mmd2(clf_out.z[batch.s == 0], clf_out.z[batch.s == 1], kernel=self.mmd_kernel)
            + binary_cross_entropy_with_logits(
                clf_out.s.squeeze(-1), batch.s, reduction="mean", weight=_iw
            )
        ) / 2
        loss = self.pred_weight * pred_loss + self.adv_weight * adv_loss

        to_log = {
            "training_clf/loss": loss,
            "training_clf/pred_loss": pred_loss,
            "training_clf/adv_loss": adv_loss,
            "training_clf/z_norm": clf_out.z.detach().norm(dim=1).mean(),
            "training_clf/z_mean_abs_diff": (
                clf_out.z[batch.s <= 0].mean() - clf_out.z[batch.s > 0].mean()
            ).abs(),
        }

        self.log_dict(to_log, logger=True)

        return loss

    @torch.no_grad()
    def threshold(self, z: Tensor) -> Tensor:
        """Go from soft to discrete features."""
        return z.sigmoid().round()

    @implements(LightningModule)
    def test_step(self, batch: Batch | CfBatch, batch_idx: int) -> ClfInferenceOut:
        assert self.built
        clf_out = self.forward(batch.x, batch.s)

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

    @implements(LightningModule)
    def test_epoch_end(self, output_results: list[ClfInferenceOut]) -> None:
        all_y = torch.cat([_r.y for _r in output_results], 0)
        all_z = torch.cat([_r.z for _r in output_results], 0)
        all_s = torch.cat([_r.s for _r in output_results], 0)
        all_preds = torch.cat([_r.preds for _r in output_results], 0)
        preds_0 = torch.cat([_r.preds_0 for _r in output_results], 0)
        preds_1 = torch.cat([_r.preds_1 for _r in output_results], 0)

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
            all_cf_y = torch.cat([_r.cf_y for _r in output_results if _r.cf_y is not None], 0)
            cf_preds = torch.cat(
                [_r.cf_preds for _r in output_results if _r.cf_preds is not None], 0
            )
            make_plot(
                x=all_cf_y.unsqueeze(-1),
                s=all_s,
                logger=self.logger,
                name="true_counterfactual_outcome",
                cols=["preds"],
            )
            make_plot(x=cf_preds, s=all_s, logger=self.logger, name="cf_preds", cols=["preds"])

    @implements(LightningModule)
    def configure_optimizers(self) -> Adam:
        return Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    @implements(CommonModel)
    def get_recon(self, dataloader: DataLoader) -> np.ndarray:
        """Should really be called get_preds."""
        preds: list[Tensor] | None = None
        for batch in dataloader:
            x = batch.x.to(self.device)
            s = batch.s.to(self.device)
            clf_out = self.forward(x, s)
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
            z, s_pred, preds = self.forward(rec, torch.ones_like(rec[:, 0]) * i)
            for _s in range(2):
                preds_dict[f"{i}_{_s}"] = (z, s_pred, preds[_s])
        return preds_dict


class ClfInferenceOut(NamedTuple):
    y: Tensor
    z: Tensor
    s: Tensor
    preds: Tensor
    preds_0: Tensor
    preds_1: Tensor
    cf_y: Tensor | None
    cf_preds: Tensor | None


class ClfFwd(NamedTuple):
    z: Tensor
    s: Tensor
    y: list[Tensor]
