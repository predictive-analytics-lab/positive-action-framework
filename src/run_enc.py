"""Run the Autoencoder."""
import logging

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger

from src.config_classes.dataclasses import Config
from src.data_modules.create import create_data_module
from src.model.encoder_model import AE

log = logging.getLogger(__name__)


def run_encoder(cfg: Config) -> None:
    """Run the X Autoencoder."""
    seed_everything(cfg.data.seed)
    data = create_data_module(cfg.data)
    data.prepare_data()
    log.info(f"data_dim={data.data_dim}, num_s={data.num_s}")
    model = AE(
        cfg.model,
        num_s=data.num_s,
        data_dim=data.data_dim,
        s_dim=data.s_dim,
        cf_available=data.cf_available,
        feature_groups=data.feature_groups,
        column_names=data.column_names,
    )
    wandb_logger = WandbLogger(
        entity="predictive-analytics-lab",
        project="aies",
        tags=cfg.training.tags.split("/")[:-1],
    )
    trainer = Trainer(max_epochs=cfg.training.epochs, logger=wandb_logger, deterministic=True)
    trainer.fit(model, datamodule=data)
    trainer.test(ckpt_path=None, datamodule=data)
