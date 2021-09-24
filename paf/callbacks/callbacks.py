from __future__ import annotations

from conduit.types import Stage
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from paf.mmd import KernelType, mmd2
from paf.plotting import make_plot

__all__ = ["L1Logger", "MmdLogger", "FeaturePlots"]


class L1Logger(Callback):
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        return self._shared(pl_module, Stage.validate)

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        return self._shared(pl_module, Stage.test)

    @staticmethod
    def _shared(pl_module: pl.LightningModule, stage: Stage) -> None:
        recon_l1 = (pl_module.all_x - pl_module.all_recon).abs().mean(dim=0)
        for i, feature_l1 in enumerate(recon_l1):
            feature_name = pl_module.data_cols[i]
            pl_module.log(
                name=f"Table6_{stage}/Ours/recon_l1 - feature {feature_name}",
                value=round(feature_l1.item(), 5),
                logger=True,
            )


class FeaturePlots(Callback):
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        return self._shared(pl_module, Stage.validate)

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        return self._shared(pl_module, Stage.test)

    @staticmethod
    def _shared(pl_module: pl.LightningModule, stage: Stage) -> None:
        if pl_module.loss.feature_groups["discrete"]:
            make_plot(
                x=pl_module.all_x[
                    :,
                    slice(
                        pl_module.loss.feature_groups["discrete"][-1].stop, pl_module.all_x.shape[1]
                    ),
                ].clone(),
                s=pl_module.all_s.clone(),
                logger=pl_module.logger,
                name=f"{stage}_true_data",
                cols=pl_module.data_cols[
                    slice(
                        pl_module.loss.feature_groups["discrete"][-1].stop, pl_module.all_x.shape[1]
                    )
                ],
                scaler=pl_module.scaler,
            )
            make_plot(
                x=pl_module.all_recon[
                    :,
                    slice(
                        pl_module.loss.feature_groups["discrete"][-1].stop, pl_module.all_x.shape[1]
                    ),
                ].clone(),
                s=pl_module.all_s.clone(),
                logger=pl_module.logger,
                name=f"{stage}_recons",
                cols=pl_module.data_cols[
                    slice(
                        pl_module.loss.feature_groups["discrete"][-1].stop, pl_module.all_x.shape[1]
                    )
                ],
                scaler=pl_module.scaler,
            )
            for group_slice in pl_module.loss.feature_groups["discrete"]:
                make_plot(
                    x=pl_module.all_x[:, group_slice].clone(),
                    s=pl_module.all_s.clone(),
                    logger=pl_module.logger,
                    name=f"{stage}_true_data",
                    cols=pl_module.data_cols[group_slice],
                    cat_plot=True,
                )
                make_plot(
                    x=pl_module.all_recon[:, group_slice].clone(),
                    s=pl_module.all_s.clone(),
                    logger=pl_module.logger,
                    name=f"{stage}_recons",
                    cols=pl_module.data_cols[group_slice],
                    cat_plot=True,
                )
        else:
            make_plot(
                x=pl_module.all_x.clone(),
                s=pl_module.all_s.clone(),
                logger=pl_module.logger,
                name=f"{stage}_true_data",
                cols=pl_module.data_cols,
            )
            make_plot(
                x=pl_module.all_recon.clone(),
                s=pl_module.all_s.clone(),
                logger=pl_module.logger,
                name=f"{stage}_recons",
                cols=pl_module.data_cols,
            )


class MmdLogger(Callback):
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        return self._shared(pl_module, Stage.validate)

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        return self._shared(pl_module, Stage.test)

    @staticmethod
    def _shared(pl_module: pl.LightningModule, stage: Stage) -> None:
        kernel = KernelType.LINEAR

        recon_mmd = mmd2(pl_module.all_x, pl_module.all_recon, kernel=kernel)
        s0_dist_mmd = mmd2(
            pl_module.all_x[pl_module.all_s == 0],
            pl_module.all_cf_pred[pl_module.all_s == 1],
            kernel=kernel,
        )
        s1_dist_mmd = mmd2(
            pl_module.all_x[pl_module.all_s == 1],
            pl_module.all_cf_pred[pl_module.all_s == 0],
            kernel=kernel,
        )

        pl_module.log(
            name=f"Logging/{stage.value}/MMD",
            value=round(recon_mmd.item(), 5),
            logger=True,
        )

        pl_module.log(
            name=f"Logging/{stage.value}/MMD S0 vs Cf",
            value=round(s0_dist_mmd.item(), 5),
            logger=True,
        )

        pl_module.log(
            name=f"Logging/{stage.value}/MMD S1 vs Cf",
            value=round(s1_dist_mmd.item(), 5),
            logger=True,
        )
