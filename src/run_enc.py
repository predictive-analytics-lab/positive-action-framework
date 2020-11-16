from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from src.config_classes.dataclasses import Config
from src.data_modules.simple_x_datamodule import SimpleXDataModule
from src.model.encoder_model import AE


def run_encoder(cfg: Config) -> None:
    data = SimpleXDataModule(cfg.data)
    data.prepare_data()
    print(data.data_dim, data.num_s)
    model = AE(cfg.model, num_s=data.num_s, data_dim=data.data_dim, s_dim=data.s_dim)
    wandb_logger = WandbLogger()
    trainer = Trainer(max_epochs=cfg.training.epochs, logger=wandb_logger)
    trainer.fit(model, datamodule=data)
    trainer.test(ckpt_path=None, datamodule=data)
