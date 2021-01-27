"""Utility functions."""
import collections
import itertools
import warnings
from typing import Any, Dict, List, MutableMapping, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from ethicml import Prediction
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import LightningLoggerBase, WandbLogger
from torch import Tensor

import wandb
from src.config_classes.dataclasses import Config
from src.data_modules.base_module import BaseDataModule
from src.logging import do_log

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


def make_plot(*, x: Tensor, s: Tensor, logger: WandbLogger, name: str, cols: List[str], cat_plot: bool = False) -> None:
    """Make plots for logging."""
    if cat_plot:
        x_df = pd.DataFrame(x.detach().cpu().numpy(), columns=cols).idxmax(axis=1).to_frame(cols[0].split("_")[0])
        cols = [cols[0].split("_")[0]]
    else:
        x_df = pd.DataFrame(x.detach().cpu().numpy(), columns=range(x.shape[1]))

    x_df["s"] = s.int().detach().cpu().numpy()

    for idx, col in enumerate(cols):
        # sns.histplot(x_df[x_df["s"] > 0][idx], kde=True, color='b')
        # sns.histplot(x_df[x_df["s"] <= 0][idx], kde=True, color='g')
        # do_log(f"histplot_image_{name}/{col}", wandb.Image(plt), logger)
        # do_log(f"histplot_plot_{name}/{col}", wandb.Plotly(plt), logger)
        # plt.clf()

        if cat_plot:
            sns.countplot(data=x_df, x=col, color='b', hue="s", palette={1: 'b', 0: 'g'})
        else:
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
        4: 7,
        5: 7,
        6: 3,
        7: 4,
        8: 7,
        9: 7,
        10: 1,
        11: 4,
        12: 7,
        13: 7,
        14: 1,
        15: 2,
        16: 8,
        17: 7,
        18: 8,
        19: 8,
        20: 8,
        21: 7,
        22: 8,
        23: 8,
        24: 8,
        25: 7,
        26: 8,
        27: 7,
        28: 8,
        29: 7,
        30: 1,
        31: 2,
    }

    preds = pd.Series({i: lookup[d] for i, d in enumerate(facct_out.hard)})

    return Prediction(hard=preds, info=facct_out.info)


def facct_mapper_2(facct_out: Prediction) -> Prediction:
    """Map from groups to outcomes."""
    lookup = {-1: 0, 1: 1, 2: 1, 3: 2, 4: 1, 5: 0, 6: 0, 7: 0, 8: 1}

    preds = pd.Series({i: lookup[d] for i, d in enumerate(facct_out.hard)})

    return Prediction(hard=preds, info=facct_out.info)


def facct_mapper_outcomes(mapped: Prediction) -> Prediction:
    """Make the final outcome."""
    lookup = {0: 0, 1: 1, 2: 0}

    preds = pd.Series({i: lookup[d] for i, d in enumerate(mapped.hard)})

    return Prediction(hard=preds, info=mapped.info)


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


def produce_selection_groups(
    data: BaseDataModule,
    outcomes: pd.DataFrame,
    recon_0: Tensor,
    recon_1: Tensor,
    logger: Optional[LightningLoggerBase],
    data_name: str = "Test",
) -> Prediction:
    """Follow Selection rules."""
    outcomes_hist(outcomes, logger)
    outcomes["decision"] = selection_rules(outcomes)
    for idx, val in outcomes["decision"].value_counts().iteritems():
        do_log(f"Table3/Ours_{data_name}/pre_selection_rule_group_{idx}", val, logger)

    _to_return = facct_mapper(Prediction(hard=outcomes["decision"]))
    for idx, val in _to_return.hard.value_counts().iteritems():
        do_log(f"Table3/Ours_{data_name}/selection_rule_group_{idx}", val, logger)

    analyse_selection_groups(data, outcomes, _to_return, recon_0, recon_1, data_name, logger)

    mapped = facct_mapper_2(_to_return)

    return facct_mapper_outcomes(mapped)


def analyse_selection_groups(
    data: BaseDataModule,
    outcomes: pd.DataFrame,
    selected: Prediction,
    recon_0: Tensor,
    recon_1: Tensor,
    data_name: str,
    logger: Optional[WandbLogger],
) -> None:
    """What's changed in these feature groups?"""
    reconstructed_0 = pd.DataFrame(recon_0.cpu().numpy(), columns=data.test_data.x.columns)
    reconstructed_1 = pd.DataFrame(recon_1.cpu().numpy(), columns=data.test_data.x.columns)

    for selection_group in [-1, 1, 2, 3, 4, 5, 6]:
        try:
            selected_data = data.test_data.x.iloc[selected.hard[selected.hard == selection_group].index]
        except IndexError:
            continue

        for group_slice in data.feature_groups["discrete"]:
            (
                reconstructed_1.iloc[selected_data.index].mean(axis=0)
                - reconstructed_0.iloc[selected_data.index].mean(axis=0)
            )[data.test_data.x.columns[group_slice]].plot(kind="bar", rot=90)
            plt.tight_layout()
            do_log(
                f"selection_group_{selection_group}_feature_groups_0-1/{data.test_data.x.columns[group_slice][0]}/{data_name}",
                wandb.Image(plt),
                logger,
            )
            plt.clf()

        for feature in data.dataset.continuous_features:
            (
                reconstructed_1.iloc[selected_data.index].mean(axis=0)
                - reconstructed_0.iloc[selected_data.index].mean(axis=0)
            )[[feature]].plot(kind="bar", rot=90)
            plt.tight_layout()
            do_log(
                f"selection_group_{selection_group}_feature_groups_0-1/{feature}/{data_name}", wandb.Image(plt), logger
            )
            plt.clf()


def outcomes_hist(outcomes: pd.DataFrame, logger: Optional[WandbLogger]) -> None:
    """Produce a distribution of the outcomes."""
    val_counts = outcomes[["s1_0_s2_0", "s1_0_s2_1", "s1_1_s2_0", "s1_1_s2_1"]].sum(axis=1).value_counts()
    sns.barplot(val_counts.index, val_counts.values)
    if logger is not None:
        logger.experiment.log({"Debugging2/Outcomes": wandb.Plotly(plt)})
        plt.clf()
    for idx, val in val_counts.iteritems():
        do_log(f"Debugging2/Ours/Outcomes-{idx}", val, logger)
    plt.clf()


def get_trainer(
    gpus: int, logger: LightningLoggerBase, max_epochs: int, callbacks: Optional[List[EarlyStopping]] = None
) -> Trainer:
    """Get a trainer object set to the right device."""
    if gpus > 0:
        return Trainer(gpus=gpus, max_epochs=max_epochs, deterministic=True, logger=logger, callbacks=callbacks)
    else:
        return Trainer(max_epochs=max_epochs, deterministic=True, logger=logger, callbacks=callbacks)


def get_wandb_logger(cfg: Config) -> Optional[WandbLogger]:
    """Get a wandb logger object."""
    if cfg.training.log:
        return WandbLogger(
            entity="predictive-analytics-lab",
            project="aies21",
            tags=cfg.training.tags.split("/")[:-1],
            config=flatten(OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)),
        )
    else:
        return None
