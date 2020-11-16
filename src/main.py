import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from src.config_classes.dataclasses import Config
from src.run_enc import run_encoder
from src.utils import flatten

cs = ConfigStore.instance()
cs.store(name="hydra", node=Config)


@hydra.main(config_name="hydra", config_path="configs")
def encoder(cfg: Config) -> None:
    args_as_dict = flatten(OmegaConf.to_container(cfg, resolve=True))  # convert to dictionary
    print("==========================\nAll args as dictionary:")
    print(args_as_dict)

    run_encoder(cfg)


if __name__ == '__main__':
    encoder()
