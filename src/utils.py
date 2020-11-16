import warnings
from pathlib import Path
from typing import List

import pandas as pd
import wandb
from ethicml import DataTuple
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from pytorch_lightning.loggers import WandbLogger
from torch import Tensor

warnings.simplefilter(action='ignore', category=FutureWarning)


def make_plot(*, x: Tensor, s: Tensor, logger: WandbLogger, name: str) -> None:

    x_df = pd.DataFrame(x.detach().cpu().numpy(), columns=range(x.shape[1]))
    x_df["s"] = s.detach().cpu().numpy()

    sns.histplot(x_df[x_df["s"] > 0][0], kde=True, color='b')
    sns.histplot(x_df[x_df["s"] <= 0][0], kde=True, color='g')
    logger.experiment.log({f"histplot_image/{name}": wandb.Image(plt)})
    logger.experiment.log({f"histplot_plot/{name}": wandb.Plotly(plt)})
    plt.clf()

    sns.distplot(x_df[x_df["s"] > 0][0], color='b')
    sns.distplot(x_df[x_df["s"] <= 0][0], color='g')
    logger.experiment.log({f"distplot_image/{name}": wandb.Image(plt)})
    plt.clf()


import collections


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
