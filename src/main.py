"""Main script."""
import logging

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from src.config_classes.dataclasses import AdultConfig, Config, SimpleXConfig
from src.run_enc import run_encoder
from src.utils import flatten

cs = ConfigStore.instance()
cs.store(name="hydra", node=Config)  # General Schema

# More specific schemas
cs.store(
    name="synth",  # name:SchemaName
    node=SimpleXConfig,  # Dataclass
    package="data",  # package:dir_within_config_path
    group="data/schema",  # group
)
cs.store(name="adult", node=AdultConfig, package="data", group="data/schema")

log = logging.getLogger(__name__)


@hydra.main(config_name="hydra", config_path="configs")
def encoder(cfg: Config) -> None:
    """Do the main encoder work."""
    args_as_dict = flatten(OmegaConf.to_container(cfg, resolve=True))  # convert to dictionary
    log.info("==========================\nAll args as dictionary:")
    log.info(args_as_dict)

    run_encoder(cfg)


if __name__ == '__main__':
    encoder()
