"""DataModule Entrypoint."""
from typing import cast

from omegaconf import OmegaConf

from src.config_classes.dataclasses import AdultConfig, DataConfig, SimpleXConfig, ThirdWayConfig
from src.data_modules.base_module import BaseDataModule
from src.data_modules.simple_adult_datamodule import SimpleAdultDataModule
from src.data_modules.simple_x_datamodule import SimpleXDataModule
from src.data_modules.third_way_datamodule import ThirdWayDataModule


def create_data_module(cfg: DataConfig) -> BaseDataModule:
    """Given a DataConfig, return a PL datamodule."""
    if OmegaConf.get_type(cfg) is AdultConfig:
        cfg = cast(AdultConfig, cfg)
        return SimpleAdultDataModule(cfg)
    elif OmegaConf.get_type(cfg) is SimpleXConfig:
        cfg = cast(SimpleXConfig, cfg)
        return SimpleXDataModule(cfg)
    elif OmegaConf.get_type(cfg) is ThirdWayConfig:
        cfg = cast(ThirdWayConfig, cfg)
        return ThirdWayDataModule(cfg)
    else:
        raise NotImplementedError("That dataset isn't prepared yet.")
