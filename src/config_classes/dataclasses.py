"""Schemas."""
from dataclasses import dataclass
from enum import Enum

from omegaconf import MISSING


@dataclass
class DataConfig:
    """base Dataset Config Schema."""

    batch_size: int = MISSING
    num_workers: int = MISSING
    seed: int = MISSING


@dataclass
class SimpleXConfig(DataConfig):
    """Simple X config schema."""

    alpha: float = MISSING
    gamma: float = MISSING
    num_samples: int = MISSING


@dataclass
class LilliputConfig(DataConfig):
    """Simple X config schema."""

    alpha: float = MISSING
    gamma: float = MISSING
    num_samples: int = MISSING


@dataclass
class ThirdWayConfig(DataConfig):
    """Third Way dataset config schema."""

    acceptance_rate: float = MISSING
    alpha: float = MISSING
    beta: float = MISSING
    gamma: float = MISSING
    num_samples: int = MISSING
    num_features: int = MISSING
    xi: float = MISSING
    num_hidden_features: int = MISSING


@dataclass
class AdultConfig(DataConfig):
    """Adult Dataset config schema."""

    bin_nationality: bool = MISSING
    bin_race: bool = MISSING
    sens: str = MISSING


class KernelType(Enum):
    """MMD Kernel Types."""

    linear = "linear"
    rbf = "rbf"


@dataclass
class ModelConfig:
    """Model Config Schema."""

    adv_blocks: int = MISSING
    encoder_blocks: int = MISSING
    decoder_blocks: int = MISSING
    latent_dims: int = MISSING
    latent_multiplier: int = MISSING
    lr: float = MISSING
    adv_weight: float = MISSING
    target_weight: float = MISSING
    reg_weight: float = MISSING  # another commmernt
    s_as_input: bool = MISSING
    mmd_kernel: KernelType = MISSING
    scheduler_rate: float = MISSING
    weight_decay: float = MISSING
    use_iw: bool = MISSING


@dataclass
class TrainingConfig:
    """Training Config Schema."""

    all_baselines: bool = MISSING
    enc_epochs: int = MISSING
    clf_epochs: int = MISSING
    tags: str = MISSING
    gpus: int = MISSING
    log: bool = MISSING


@dataclass
class Config:
    """Base Config Schema."""

    data: DataConfig = MISSING
    enc: ModelConfig = MISSING  # put config files for this into `conf/model/`
    clf: ModelConfig = MISSING  # put config files for this into `conf/model/`
    training: TrainingConfig = MISSING
