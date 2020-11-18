"""Schemas."""
from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class DataConfig:
    """base Dataset Config Schema."""

    batch_size: int = MISSING
    dataset: str = MISSING
    num_workers: int = MISSING
    seed: int = MISSING


@dataclass
class SimpleXConfig(DataConfig):
    """Simple X config schema."""

    alpha: float = MISSING
    gamma: float = MISSING
    num_samples: int = MISSING


@dataclass
class ThirdWayConfig(DataConfig):
    """Third Way dataset config schema."""

    alpha: float = MISSING
    gamma: float = MISSING
    num_samples: int = MISSING


@dataclass
class AdultConfig(DataConfig):
    """Adult Dataset config schema."""

    bin_nationality: bool = MISSING


@dataclass
class ModelConfig:
    """Model Config Schema."""

    blocks: int = MISSING
    latent_dims: int = MISSING
    latent_multiplier: int = MISSING
    lr: float = MISSING
    adv_weight: float = MISSING
    recon_weight: float = MISSING
    reg_weight: float = MISSING  # another commmernt
    s_as_input: bool = MISSING


@dataclass
class TrainingConfig:
    """Training Config Schema."""

    epochs: int = MISSING
    tags: str = MISSING


@dataclass
class Config:
    """Base Config Schema."""

    data: DataConfig = MISSING
    model: ModelConfig = MISSING  # put config files for this into `conf/model/`
    training: TrainingConfig = MISSING
