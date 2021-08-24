from __future__ import annotations

from bolts.structures import Stage
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from paf.config_classes.dataclasses import KernelType
from paf.log_progress import do_log
from paf.mmd import mmd2


class MseLogger(Callback):
    def __init__(self) -> None:
        super().__init__()

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        recon_mse = (pl_module.all_x - pl_module.all_recon).abs().mean(dim=0)
        for i, feature_mse in enumerate(recon_mse):
            feature_name = pl_module.data_cols[i]
            pl_module.log(
                name=f"Table6/Ours/recon_l1 - feature {feature_name}",
                value=round(feature_mse.item(), 5),
                logger=True,
            )


class MmdLogger(Callback):
    def __init__(self) -> None:
        super().__init__()

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        return self._shared(pl_module, Stage.validate)

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        return self._shared(pl_module, Stage.test)

    def _shared(self, pl_module: pl.LightningModule, stage: Stage) -> None:
        kernel = KernelType.linear

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
