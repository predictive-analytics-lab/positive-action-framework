# Generated by configen, do not edit.
# See https://github.com/facebookresearch/hydra/tree/master/tools/configen
# fmt: off
# isort:skip_file
# flake8: noqa

from dataclasses import dataclass
from omegaconf import MISSING
from paf.config_classes.dataclasses import KernelType


@dataclass
class AEConf:
    _target_: str = "paf.model.AE"
    s_as_input: bool = MISSING
    latent_dims: int = MISSING
    encoder_blocks: int = MISSING
    latent_multiplier: int = MISSING
    adv_blocks: int = MISSING
    decoder_blocks: int = MISSING
    adv_weight: float = MISSING
    cycle_weight: float = MISSING
    target_weight: float = MISSING
    lr: float = MISSING
    mmd_kernel: KernelType = MISSING
    scheduler_rate: float = MISSING
    weight_decay: float = MISSING


@dataclass
class ClfConf:
    _target_: str = "paf.model.Clf"
    adv_weight: float = MISSING
    reg_weight: float = MISSING
    pred_weight: float = MISSING
    lr: float = MISSING
    s_as_input: bool = MISSING
    latent_dims: int = MISSING
    mmd_kernel: KernelType = MISSING
    scheduler_rate: float = MISSING
    weight_decay: float = MISSING
    use_iw: bool = MISSING
    encoder_blocks: int = MISSING
    adv_blocks: int = MISSING
    decoder_blocks: int = MISSING
    latent_multiplier: int = MISSING


@dataclass
class CycleGanConf:
    _target_: str = "paf.model.CycleGan"
    d_lr: float = 0.0002
    g_lr: float = 0.0002
    beta_1: float = 0.5
    beta_2: float = 0.999
    epoch_decay: int = 200
