from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class DataConfig:
    alpha: float = MISSING
    batch_size: int = MISSING
    gamma: float = MISSING
    num_samples: int = MISSING
    num_workers: int = MISSING
    seed: int = MISSING


@dataclass
class ModelConfig:
    blocks: int = MISSING
    latent_dims: int = MISSING
    latent_multiplier: int = MISSING
    lr: float = MISSING
    recon_weight: float = MISSING
    reg_weight: float = MISSING  # another commmernt
    s_as_input: bool = MISSING


@dataclass
class TrainingConfig:
    epochs: int = MISSING


@dataclass
class Config:  # base config schema
    data: DataConfig = MISSING
    model: ModelConfig = MISSING  # put config files for this into `conf/model/`
    training: TrainingConfig = MISSING
