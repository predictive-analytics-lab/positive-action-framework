"""Plotting related functions."""
from typing import List, Optional

import matplotlib as mpl
import pandas as pd
import seaborn as sns
import wandb
from ethicml import DataTuple
from matplotlib import pyplot as plt
from pytorch_lightning.loggers import WandbLogger
from torch import Tensor

from paf.logging_i_guess import do_log


def label_plot(data: DataTuple, logger: Optional[WandbLogger], name: str = ""):
    """Make a label (quadrant) plot and uplad to wandb."""
    s_col = data.s.columns[0]
    s_values = data.s[s_col].value_counts() / data.s[s_col].count()
    if len(s_values) == 1:
        missing = s_values.index.min()
        missing_val = s_values[missing]
        s_values[1 - missing] = 1 - missing_val

    s_0_val = s_values[0]
    s_1_val = s_values[1]

    s_0_label = s_values.index.min()
    s_1_label = s_values.index.max()

    y_col = data.y.columns[0]
    y_s0 = (
        data.y[y_col][data.s[s_col] == 0].value_counts() / data.y[y_col][data.s[s_col] == 0].count()
    )
    y_s1 = (
        data.y[y_col][data.s[s_col] == 1].value_counts() / data.y[y_col][data.s[s_col] == 1].count()
    )

    if len(y_s1) == 0:
        y_s1[0] = 0
        y_s1[1] = 1
    if len(y_s0) == 0:
        y_s0[0] = 0
        y_s0[1] = 1

    y_0_label = y_s0.index[0]
    y_1_label = y_s0.index[1]

    mpl.style.use("seaborn-pastel")
    # plt.xkcd()

    fig, plot = plt.subplots()

    quadrant1 = plot.bar(
        0,
        height=y_s0[y_0_label] * 100,
        width=s_0_val * 100,
        align="edge",
        edgecolor="black",
        color="C0",
    )
    quadrant2 = plot.bar(
        s_0_val * 100,
        height=y_s1[y_0_label] * 100,
        width=s_1_val * 100,
        align="edge",
        edgecolor="black",
        color="C1",
    )
    quadrant3 = plot.bar(
        0,
        height=y_s0[y_1_label] * 100,
        width=s_0_val * 100,
        bottom=y_s0[y_0_label] * 100,
        align="edge",
        edgecolor="black",
        color="C2",
    )
    quadrant4 = plot.bar(
        s_0_val * 100,
        height=y_s1[y_1_label] * 100,
        width=s_1_val * 100,
        bottom=y_s1[y_0_label] * 100,
        align="edge",
        edgecolor="black",
        color="C3",
    )

    plot.set_ylim(0, 100)
    plot.set_xlim(0, 100)
    plot.set_ylabel(f"Percent {y_col}=y")
    plot.set_xlabel(f"Percent {s_col}=s")
    plot.set_title("Dataset Composition by class and sensitive attribute")

    plot.legend(
        [quadrant1, quadrant2, quadrant3, quadrant4],
        [
            f"y={y_0_label}, s={s_0_label}",
            f"y={y_0_label}, s={s_1_label}",
            f"y={y_1_label}, s={s_0_label}",
            f"y={y_1_label}, s={s_1_label}",
        ],
    )

    do_log(f"label_plot/{name}", wandb.Image(plt), logger)

    plt.clf()


def make_plot(
    *, x: Tensor, s: Tensor, logger: WandbLogger, name: str, cols: List[str], cat_plot: bool = False
) -> None:
    """Make plots for logging."""
    if cat_plot:
        x_df = (
            pd.DataFrame(x.detach().cpu().numpy(), columns=cols)
            .idxmax(axis=1)
            .to_frame(cols[0].split("_")[0])
        )
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


def outcomes_hist(outcomes: pd.DataFrame, logger: Optional[WandbLogger]) -> None:
    """Produce a distribution of the outcomes."""
    val_counts = (
        outcomes[["s1_0_s2_0", "s1_0_s2_1", "s1_1_s2_0", "s1_1_s2_1"]].sum(axis=1).value_counts()
    )
    sns.barplot(val_counts.index, val_counts.values)
    if logger is not None:
        logger.experiment.log({"Debugging2/Outcomes": wandb.Image(plt)})
        plt.clf()
    for idx, val in val_counts.iteritems():
        do_log(f"Debugging2/Ours/Outcomes-{idx}", val, logger)
    plt.clf()
