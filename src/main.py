"""Main script."""
import logging
from typing import Final

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from src.config_classes.dataclasses import AdultConfig, Config, SimpleXConfig, ThirdWayConfig
from src.run_model import run_aies
from src.utils import flatten

cs = ConfigStore.instance()
cs.store(name="hydra", node=Config)  # General Schema

# More specific Schemas, each has a name and a deifnition (node)
data_package: Final[str] = "data"  # package:dir_within_config_path
data_group: Final[str] = "data/schema"  # group
cs.store(name="adult", node=AdultConfig, package=data_package, group=data_group)
cs.store(name="synth", node=SimpleXConfig, package=data_package, group=data_group)
cs.store(name="third", node=ThirdWayConfig, package=data_package, group=data_group)

log = logging.getLogger(__name__)


@hydra.main(config_name="hydra", config_path="configs")
def aies(cfg: Config) -> None:
    """Do the main encoder work."""
    args_as_dict = flatten(OmegaConf.to_container(cfg, resolve=True))  # convert to dictionary
    log.info("==========================\nAll args as dictionary:")
    log.info(args_as_dict)

    run_aies(cfg)


if __name__ == '__main__':
    aies()
