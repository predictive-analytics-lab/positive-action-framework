# Generated by configen, do not edit.
# See https://github.com/facebookresearch/hydra/tree/master/tools/configen
# fmt: off
# isort:skip_file
# flake8: noqa

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LRCVConf:
    _target_: str = "ethicml.LRCV"
    n_splits: int = 3
    seed: int = 888


@dataclass
class KamiranConf:
    _target_: str = "ethicml.Kamiran"
    classifier: str = "LR"
    C: Optional[float] = None
    kernel: Optional[str] = None
    seed: int = 888


@dataclass
class OracleConf:
    _target_: str = "ethicml.Oracle"


@dataclass
class DPOracleConf:
    _target_: str = "ethicml.DPOracle"


@dataclass
class ZafarFairnessConf:
    _target_: str = "ethicml.ZafarFairness"
    c: float = 0.001


@dataclass
class KamishimaConf:
    _target_: str = "ethicml.Kamishima"
    eta: float = 1.0


@dataclass
class AgarwalConf:
    _target_: str = "ethicml.Agarwal"
    fairness: str = "DP"
    classifier: str = "LR"
    eps: float = 0.1
    iters: int = 50
    C: Optional[float] = None
    kernel: Optional[str] = None
    seed: int = 888
