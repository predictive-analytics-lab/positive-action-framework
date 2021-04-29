"""DataModule Entrypoint."""
from typing import cast

from omegaconf import OmegaConf

from paf.base_templates.base_module import BaseDataModule
from paf.config_classes.dataclasses import (
    AdultConfig,
    DataConfig,
    LilliputConfig,
    SimpleXConfig,
    ThirdWayConfig,
)
from paf.data_modules.lilliput_datamodule import LilliputDataModule
from paf.data_modules.simple_adult_datamodule import SimpleAdultDataModule
from paf.data_modules.simple_x_datamodule import SimpleXDataModule
from paf.data_modules.third_way_datamodule import ThirdWayDataModule


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
    elif OmegaConf.get_type(cfg) is LilliputConfig:
        cfg = cast(LilliputConfig, cfg)
        return LilliputDataModule(cfg)
    else:
        raise NotImplementedError("That dataset isn't prepared yet.")