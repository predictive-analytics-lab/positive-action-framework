"""Selection process."""

import itertools
from typing import Optional

from ethicml import Prediction
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pytorch_lightning.loggers import LightningLoggerBase, WandbLogger
import seaborn as sns
from torch import Tensor
import wandb

from paf.base_templates.base_module import BaseDataModule
from paf.log_progress import do_log
from paf.plotting import outcomes_hist
from paf.utils import facct_mapper, facct_mapper_2, facct_mapper_outcomes


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
    outcomes: pd.DataFrame,
    data: Optional[BaseDataModule] = None,
    recon_0: Optional[Tensor] = None,
    recon_1: Optional[Tensor] = None,
    logger: Optional[LightningLoggerBase] = None,
    data_name: str = "Test",
    fair: bool = False,
) -> Prediction:
    """Follow Selection rules."""
    outcomes_hist(outcomes, logger)
    outcomes["decision"] = selection_rules(outcomes)
    for idx, val in outcomes["decision"].value_counts().iteritems():
        do_log(f"Table3/Ours_{data_name}/pre_selection_rule_group_{idx}", val, logger)

    if recon_1 is not None:
        assert data is not None
        analyse_selection_groups(
            data=data,
            outcomes=outcomes,
            selected=Prediction(hard=outcomes["decision"]),
            recon_0=recon_0,
            recon_1=recon_1,
            data_name=f"PreSelection_{data_name}",
            logger=logger,
        )

    _to_return = facct_mapper(Prediction(hard=outcomes["decision"]))
    for idx, val in _to_return.hard.value_counts().iteritems():
        do_log(f"Table3/Ours_{data_name}/selection_rule_group_{idx}", val, logger)

    if recon_1 is not None:
        assert data is not None
        analyse_selection_groups(
            data=data,
            outcomes=outcomes,
            selected=_to_return,
            recon_0=recon_0,
            recon_1=recon_1,
            data_name=data_name,
            logger=logger,
        )

    mapped = facct_mapper_2(_to_return)

    return facct_mapper_outcomes(mapped, fair)


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

    for selection_group in range(selected.hard.min(), selected.hard.max()):
        try:
            selected_data = data.test_data.x.iloc[
                selected.hard[selected.hard == selection_group].index
            ]
        except IndexError:
            continue

        for group_slice in data.feature_groups["discrete"]:
            if selection_group % 2 == 0:
                (
                    reconstructed_0.iloc[selected_data.index].sum(axis=0)
                    - reconstructed_1.iloc[selected_data.index].sum(axis=0)
                )[data.test_data.x.columns[group_slice]].plot(kind="bar", rot=90)
            else:
                (
                    reconstructed_1.iloc[selected_data.index].sum(axis=0)
                    - reconstructed_0.iloc[selected_data.index].sum(axis=0)
                )[data.test_data.x.columns[group_slice]].plot(kind="bar", rot=90)
            plt.xticks(rotation=90)
            plt.tight_layout()
            do_log(
                f"{data_name}_selection_group_{selection_group}_feature_groups_0-1/{data.test_data.x.columns[group_slice][0].split('_')[0]}",
                wandb.Image(plt),
                logger,
            )
            plt.clf()

        for feature in data.dataset.continuous_features:

            sns.distplot(reconstructed_1.iloc[selected_data.index][feature], color='b')
            sns.distplot(reconstructed_0.iloc[selected_data.index][feature], color='g')
            # (
            #     reconstructed_1.iloc[selected_data.index].mean(axis=0)
            #     - reconstructed_0.iloc[selected_data.index].mean(axis=0)
            # )[[feature]].plot(kind="bar", rot=90)
            plt.xticks(rotation=90)
            plt.tight_layout()
            do_log(
                f"{data_name}_selection_group_{selection_group}_feature_groups_0-1/{feature}",
                wandb.Image(plt),
                logger,
            )
            plt.clf()
