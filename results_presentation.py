"""Small script that takes a results csv d/l form W&B and makes it pretty looking."""
import math
from pathlib import Path

import numpy as np
import pandas as pd
import typer


def cell(data):
    """Make mean and std have +-."""
    mean = round(np.mean(data), 3)
    if math.isnan(mean):
        return "N/A"
    return f"{mean:.3f} $\\pm$ {round(np.std(data), 3):.3f}"


def main(raw_csv: Path):
    """Run on results to get table."""
    data = pd.read_csv(raw_csv)
    data = data.drop(["Name"], axis=1)

    data.columns = pd.MultiIndex.from_tuples(
        [col.split('/') for col in data.columns], names=["group", "model", "metric"]
    )

    data = data.T

    data = data.groupby(level=[0, 1, 2]).agg(cell)

    data = data.reset_index().pivot(index=["group", "model"], columns="metric")

    data.columns = data.columns.droplevel()

    data = data[
        [
            "P(Y=1|S=0)",
            "P(Y=1|S=1)",
            "P(Ty=1|S=0)",
            "P(Ty=1|S=1)",
            'P(Ty=1|S=0,Y=1)',
            'P(Ty=1|S=1,Y=1)',
            'P(Ty=1|S=0,Y=0)',
            'P(Ty=1|S=1,Y=0)',
            'P(Ty=0|S=0,Y=1)',
            'P(Ty=0|S=1,Y=1)',
            'Accuracy',
            'TPR-sens_0',
            'TPR-sens_1',
            'TNR-sens_0',
            'TNR-sens_1',
            'prob_pos-sens_0',
            'prob_pos-sens_1',
        ]
    ]

    data.to_csv('./show.csv')


if __name__ == "__main__":
    typer.run(main)
