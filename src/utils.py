"""Utility functions."""
import collections
import itertools
import logging
import warnings
from typing import Any, Dict, List, MutableMapping, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from ethicml import Prediction
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import LightningLoggerBase, WandbLogger
from torch import Tensor

import wandb
from src.config_classes.dataclasses import Config

log = logging.getLogger(__name__)


warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


def make_plot(*, x: Tensor, s: Tensor, logger: WandbLogger, name: str, cols: List[str]) -> None:
    """Make plots for logging."""
    x_df = pd.DataFrame(x.detach().cpu().numpy(), columns=range(x.shape[1]))
    x_df["s"] = s.detach().cpu().numpy()

    for idx, col in enumerate(cols):
        # sns.histplot(x_df[x_df["s"] > 0][idx], kde=True, color='b')
        # sns.histplot(x_df[x_df["s"] <= 0][idx], kde=True, color='g')
        # do_log(f"histplot_image_{name}/{col}", wandb.Image(plt), logger)
        # do_log(f"histplot_plot_{name}/{col}", wandb.Plotly(plt), logger)
        # plt.clf()

        sns.distplot(x_df[x_df["s"] > 0][idx], color='b')
        sns.distplot(x_df[x_df["s"] <= 0][idx], color='g')
        do_log(f"distplot_image_{name}/{col}", wandb.Image(plt), logger)
        plt.clf()


def flatten(d: MutableMapping[Any, Any], parent_key: str = "", sep: str = ".") -> Dict[Any, Any]:
    """Flatten a nested dictionary by separating the keys with `sep`."""
    items: List[Any] = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def facct_mapper(facct_out: Prediction) -> Prediction:
    """Map from groups to outcomes."""
    lookup = {
        0: 5,
        1: 6,
        2: 3,
        3: 4,
        4: -1,
        5: -1,
        6: 3,
        7: 4,
        8: -1,
        9: -1,
        10: 1,
        11: 4,
        12: -1,
        13: -1,
        14: 1,
        15: 2,
        16: -1,
        17: -1,
        18: -1,
        19: -1,
        20: -1,
        21: -1,
        22: -1,
        23: -1,
        24: -1,
        25: -1,
        26: -1,
        27: -1,
        28: -1,
        29: -1,
        30: 1,
        31: 2,
    }

    preds = pd.Series({i: lookup[d] for i, d in enumerate(facct_out.hard)})

    return Prediction(hard=preds, info=facct_out.info)


def facct_mapper_2(facct_out: Prediction) -> Prediction:
    """Map from groups to outcomes."""
    lookup = {-1: 0, 1: 1, 2: 1, 3: 0, 4: 1, 5: 0, 6: 0}

    preds = pd.Series({i: lookup[d] for i, d in enumerate(facct_out.hard)})

    return Prediction(hard=preds, info=facct_out.info)


def selection_rules(outcome_df: pd.DataFrame) -> np.ndarray:
    """Apply selection rules."""
    conditions = [
        (outcome_df['s1_0_s2_0'] == a)
        & (outcome_df['s1_0_s2_1'] == b)
        & (outcome_df['s1_1_s2_0'] == c)
        & (outcome_df['s1_1_s2_1'] == d)
        & (outcome_df['true_s'] == e)
        for a, b, c, d, e in itertools.product([0, 1], repeat=5)
    ]

    values = list(range(len(conditions)))

    return np.select(conditions, values, -1)


def do_log(name: str, val: Any, logger: Optional[WandbLogger]) -> None:
    """Log to experiment tracker and also the logger."""
    if isinstance(val, float) or isinstance(val, int):
        log.info(f"{name}: {val}")
    if logger is not None:
        logger.experiment.log({name: val})


def produce_selection_groups(
    outcomes: pd.DataFrame, logger: Optional[LightningLoggerBase], data: str = "Test"
) -> Prediction:
    """Follow Selection rules."""
    outcomes_hist(outcomes, logger)
    outcomes["decision"] = selection_rules(outcomes)
    for idx, val in outcomes["decision"].value_counts().iteritems():
        do_log(f"Table3/Ours_{data}/pre_selection_rule_group_{idx}", val, logger)

    _to_return = facct_mapper(Prediction(hard=outcomes["decision"]))
    for idx, val in _to_return.hard.value_counts().iteritems():
        do_log(f"Table3/Ours_{data}/selection_rule_group_{idx}", val, logger)

    to_return = facct_mapper_2(_to_return)
    return to_return


def outcomes_hist(outcomes: pd.DataFrame, logger: Optional[WandbLogger]) -> None:
    """Produce a distribution of the outcomes."""
    val_counts = outcomes[["s1_0_s2_0", "s1_0_s2_1", "s1_1_s2_0", "s1_1_s2_1"]].sum(axis=1).value_counts()
    sns.barplot(val_counts.index, val_counts.values)
    if logger is not None:
        logger.experiment.log({"Debugging2/Outcomes": wandb.Plotly(plt)})
    for idx, val in val_counts.iteritems():
        do_log(f"Debugging2/Ours/Outcomes-{idx}", val, logger)
    plt.clf()


def get_trainer(gpus: int, logger: LightningLoggerBase, max_epochs: int) -> Trainer:
    """Get a trainer object set to the right device."""
    if gpus > 0:
        return Trainer(gpus=gpus, max_epochs=max_epochs, deterministic=True, logger=logger)
    else:
        return Trainer(max_epochs=max_epochs, deterministic=True, logger=logger)


def get_wandb_logger(cfg: Config) -> WandbLogger:
    """Get a wandb logger object."""
    return WandbLogger(
        entity="predictive-analytics-lab",
        project="aies21",
        tags=cfg.training.tags.split("/")[:-1],
        config=flatten(OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)),
    )
