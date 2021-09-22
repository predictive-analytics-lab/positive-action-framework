"""Small script that takes a results csv d/l form W&B and makes it pretty looking."""
from __future__ import annotations
from pathlib import Path

import pandas as pd
import typer


def main(raw_csv: Path) -> None:
    """Run on results to get table."""
    data = pd.read_csv(raw_csv)
    data = data.drop(["Name"] + [col for col in data.columns if "val" in col], axis=1)

    data = pd.DataFrame(
        [
            {
                a: f"{b*100:.4f} +/- {d*100:.4f}"
                for (a, b), (c, d) in zip(data.mean().iteritems(), data.std().iteritems())
            }
        ]
    )

    data.columns = pd.MultiIndex.from_tuples(
        [col.split('/') for col in data.columns], names=("group", "model", "metric")
    )

    data = data.T

    data = data.reset_index().pivot(index=["group", "model"], columns="metric")

    data.columns = data.columns.droplevel()
    data = data[
        [
            "P(Y=1|S=0)",
            "P(Y=1|S=1)",
            "P(Y=1|S=0,Ty=1)",
            "P(Y=1|S=1,Ty=1)",
            "P(Y=1|S=0,Ty=0)",
            "P(Y=1|S=1,Ty=0)",
            "P(Y=0|S=0,Ty=0)",
            "P(Y=0|S=1,Ty=0)",
            "P(Ty=1|S=0)",
            "P(Ty=1|S=1)",
            "P(Ty=1|Y=1)",
            "P(Ty=1|Y=0)",
            "P(Ty=0|Y=1)",
            "P(Ty=0|Y=0)",
            "P(Ty=0)",
            "P(Ty=1)",
            "P(Y=0)",
            "P(Y=1)",
            'P(Ty=1|S=0,Y=1)',
            'P(Ty=1|S=1,Y=1)',
            'P(Ty=1|S=0,Y=0)',
            'P(Ty=1|S=1,Y=0)',
            'P(Ty=0|S=0,Y=0)',
            'P(Ty=0|S=0,Y=1)',
            'P(Ty=0|S=1,Y=0)',
            'P(Ty=0|S=1,Y=1)',
            'Accuracy',
            'TPR-sens_0',
            'TPR-sens_1',
            'TNR-sens_0',
            'TNR-sens_1',
            'prob_pos-sens_0',
            'prob_pos-sens_1',
            "FID",
            "prob_pos-Abs-Diff-sens_0-sens_1",
            # "MMD",
            # "MMD S0 vs Cf",
            # "MMD S1 vs Cf",
        ]
        + [
            f"pre_selection_rule_group_{i}"
            for i in range(32)
            if f"pre_selection_rule_group_{i}" in data.columns
        ]
    ]

    data.dropna(axis=0, how="all").transpose().to_csv('./show.csv')


if __name__ == "__main__":
    typer.run(main)


def test():
    main(Path("~/Downloads/wandb_export_2021-08-26T17_05_06.746+01_00.csv"))
