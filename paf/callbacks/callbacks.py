from __future__ import annotations

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from paf.log_progress import do_log


class MseLogger(Callback):
    def __init__(self):
        super().__init__()

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        recon_mse = (pl_module.all_x - pl_module.all_recon).abs().mean(dim=0)
        for i, feature_mse in enumerate(recon_mse):
            feature_name = pl_module.data_cols[i]
            do_log(
                f"Table6/Ours/recon_l1 - feature {feature_name}",
                round(feature_mse.item(), 5),
                pl_module.logger,
            )
