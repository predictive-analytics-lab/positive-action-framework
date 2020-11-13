from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class DataConfig:
    alpha: float = MISSING
    gamma: float = MISSING
    num_samples: int = MISSING
    seed: int = MISSING


@dataclass
class ModelConfig:
    blocks: int = MISSING
    latent_dims: int = MISSING
    latent_multiplier: int = MISSING
    lr: float = MISSING
    recon_weight: float = MISSING
    reg_weight: float = MISSING  # another commmernt


@dataclass
class TrainingConfig:
    epochs: int = MISSING


@dataclass
class Config:  # base config schema
    data: DataConfig = MISSING
    model: ModelConfig = MISSING  # put config files for this into `conf/model/`
    training: TrainingConfig = MISSING
