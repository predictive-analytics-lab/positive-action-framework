"""Utility functions."""
import collections
import warnings
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pytorch_lightning.loggers import WandbLogger
from torch import Tensor

import wandb

warnings.simplefilter(action='ignore', category=FutureWarning)


def make_plot(*, x: Tensor, s: Tensor, logger: WandbLogger, name: str, cols: List[str]) -> None:
    """Make plots for logging."""
    x_df = pd.DataFrame(x.detach().cpu().numpy(), columns=range(x.shape[1]))
    x_df["s"] = s.detach().cpu().numpy()

    for idx, col in enumerate(cols):
        sns.histplot(x_df[x_df["s"] > 0][idx], kde=True, color='b')
        sns.histplot(x_df[x_df["s"] <= 0][idx], kde=True, color='g')
        logger.experiment.log({f"histplot_image_{name}/{col}": wandb.Image(plt)})
        logger.experiment.log({f"histplot_plot_{name}/{col}": wandb.Plotly(plt)})
        plt.clf()

        sns.distplot(x_df[x_df["s"] > 0][idx], color='b')
        sns.distplot(x_df[x_df["s"] <= 0][idx], color='g')
        logger.experiment.log({f"distplot_image_{name}/{col}": wandb.Image(plt)})
        plt.clf()


def flatten(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """Flatten a nested dictionary by separating the keys with `sep`."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
