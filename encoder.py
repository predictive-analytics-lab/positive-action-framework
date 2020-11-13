from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from config_classes.dataclasses import Config
from data_modules import SimpleXDataModule
from model.encoder_model import AE


def run_encoder(cfg: Config):
    data = SimpleXDataModule(cfg.data)
    model = AE(cfg.model, data.num_s)
    wandb_logger = WandbLogger()
    trainer = Trainer(max_epochs=cfg.training.epochs, logger=wandb_logger)
    trainer.fit(model, data)
    trainer.test(ckpt_path=None)
