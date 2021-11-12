"""Selection process."""
from __future__ import annotations
import itertools

from ethicml import Prediction
from matplotlib import pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import pytorch_lightning.loggers as pll
import seaborn as sns
from torch import Tensor
from typing_extensions import Final

from paf.base_templates.base_module import BaseDataModule
from paf.log_progress import do_log
from paf.utils import facct_mapper, facct_mapper_2, facct_mapper_outcomes
import wandb

GROUP_0: Final[str] = "initial_group"
GROUP_1: Final[str] = "first_grouping"
GROUP_2: Final[str] = "second_grouping"
GROUP_3: Final[str] = "decision"


def selection_rules(outcome_df: pd.DataFrame) -> npt.NDArray[np.int_]:
    """Apply selection rules."""
    conditions = [
        (outcome_df["s1_0_s2_0"] == a)
        & (outcome_df["s1_0_s2_1"] == b)
        & (outcome_df["s1_1_s2_0"] == c)
        & (outcome_df["s1_1_s2_1"] == d)
        & (outcome_df["true_s"] == e)
        for a, b, c, d, e in itertools.product([0, 1], repeat=5)
    ]

    values = list(range(len(conditions)))

    return np.select(conditions, values, -1)


def baseline_selection_rules(
    outcomes: pd.DataFrame, *, data_name: str, logger: pll.LightningLoggerBase | None, fair: bool
) -> Prediction:
    conditions = [
        (outcomes["s1_0_s2_0"] == 0) & (outcomes["s1_1_s2_1"] == 0) & (outcomes["true_s"] == 0),
        (outcomes["s1_0_s2_0"] == 0) & (outcomes["s1_1_s2_1"] == 0) & (outcomes["true_s"] == 1),
        (outcomes["s1_0_s2_0"] == 0) & (outcomes["s1_1_s2_1"] == 1) & (outcomes["true_s"] == 0),
        (outcomes["s1_0_s2_0"] == 0) & (outcomes["s1_1_s2_1"] == 1) & (outcomes["true_s"] == 1),
        (outcomes["s1_0_s2_0"] == 1) & (outcomes["s1_1_s2_1"] == 0) & (outcomes["true_s"] == 0),
        (outcomes["s1_0_s2_0"] == 1) & (outcomes["s1_1_s2_1"] == 0) & (outcomes["true_s"] == 1),
        (outcomes["s1_0_s2_0"] == 1) & (outcomes["s1_1_s2_1"] == 1) & (outcomes["true_s"] == 0),
        (outcomes["s1_0_s2_0"] == 1) & (outcomes["s1_1_s2_1"] == 1) & (outcomes["true_s"] == 1),
    ]

    values = [0, 1, 2, 3, 4, 5, 6, 7]
    outcomes[GROUP_1] = np.select(conditions, values, -1)

    lookup = {0: 0, 1: 0, 2: 2, 3: 0, 4: 1, 5: 1, 6: 1, 7: 1}
    outcomes[GROUP_2] = pd.Series({i: lookup[d] for i, d in enumerate(outcomes[GROUP_1])})

    outcomes[GROUP_3] = facct_mapper_outcomes(pd.Series(outcomes[GROUP_2]), fair=fair)
    if logger is not None:
        logger.experiment.log(
            {
                f"Groups/{data_name}/{group}/{key}": value
                for group in [GROUP_1, GROUP_2, GROUP_3]
                for key, value in outcomes[group].value_counts().items()
            }
        )
    return Prediction(hard=pd.Series(outcomes[GROUP_3]))


def produce_selection_groups(
    outcomes: pd.DataFrame,
    *,
    data_name: str,
    data: BaseDataModule | None = None,
    recon_0: Tensor | None = None,
    recon_1: Tensor | None = None,
    logger: pll.WandbLogger | None = None,
    fair: bool = False,
    debug: bool = False,
) -> Prediction:
    """Follow Selection rules."""
    # if logger is not None:
    #     outcomes_hist(outcomes, logger)
    outcomes[GROUP_0] = selection_rules(outcomes)

    # if recon_1 is not None:
    #     assert recon_0 is not None
    #     assert recon_1 is not None
    #     assert data is not None
    #     analyse_selection_groups(
    #         data=data,
    #         selected=Prediction(hard=outcomes["decision"]),
    #         recon_0=recon_0,
    #         recon_1=recon_1,
    #         data_name=f"PreSelection_{data_name}",
    #         logger=logger,
    #     )

    outcomes[GROUP_1] = facct_mapper(pd.Series(outcomes[GROUP_0]))
    print(outcomes[GROUP_1].value_counts())

    outcomes[GROUP_2] = facct_mapper_2(pd.Series(outcomes[GROUP_1]))

    outcomes[GROUP_3] = facct_mapper_outcomes(pd.Series(outcomes[GROUP_2]), fair=fair)

    if recon_1 is not None and debug:
        assert recon_0 is not None
        assert recon_1 is not None
        assert data is not None
        analyse_selection_groups(
            data=data,
            selected=Prediction(hard=pd.Series(outcomes[GROUP_1])),
            recon_0=recon_0,
            recon_1=recon_1,
            data_name=data_name,
            logger=logger,
        )

    if logger is not None:
        logger.experiment.log(
            {
                f"Groups/{data_name}/{group}/{key}": value
                for group in [GROUP_0, GROUP_1, GROUP_2, GROUP_3]
                for key, value in outcomes[group].value_counts().items()
            }
        )
    return Prediction(hard=pd.Series(outcomes[GROUP_3]))


def analyse_selection_groups(
    data: BaseDataModule,
    selected: Prediction,
    recon_0: Tensor,
    recon_1: Tensor,
    data_name: str,
    logger: pll.WandbLogger,
) -> None:
    """What's changed in these feature groups?"""
    reconstructed_0 = pd.DataFrame(recon_0.cpu().numpy(), columns=data.test_datatuple.x.columns)
    reconstructed_1 = pd.DataFrame(recon_1.cpu().numpy(), columns=data.test_datatuple.x.columns)

    for selection_group in range(selected.hard.min(), selected.hard.max() + 1):
        try:
            selected_data = data.test_datatuple.x.iloc[
                selected.hard[selected.hard == selection_group].index  # type: ignore[call-overload]
            ]
        except IndexError:
            continue

        for group_slice in data.feature_groups["discrete"]:
            (
                reconstructed_0.iloc[selected_data.index].sum(axis=0)
                - reconstructed_1.iloc[selected_data.index].sum(axis=0)
            )[data.test_datatuple.x.columns[group_slice]].plot(kind="bar", rot=90)
            plt.xticks(rotation=90)
            plt.tight_layout()
            do_log(
                f"{data_name}_selection_group_{selection_group}_feature_groups_0-1"
                f"/{data.test_datatuple.x.columns[group_slice][0].split('_')[0]}",
                wandb.Image(plt),
                logger,
            )
            plt.clf()
            (
                reconstructed_1.iloc[selected_data.index].sum(axis=0)
                - reconstructed_0.iloc[selected_data.index].sum(axis=0)
            )[data.test_datatuple.x.columns[group_slice]].plot(kind="bar", rot=90)
            plt.xticks(rotation=90)
            plt.tight_layout()
            do_log(
                f"{data_name}_selection_group_{selection_group}_feature_groups_1-0"
                f"/{data.test_datatuple.x.columns[group_slice][0].split('_')[0]}",
                wandb.Image(plt),
                logger,
            )
            plt.clf()

        for feature in data.cont_features:

            sns.distplot(reconstructed_1.iloc[selected_data.index][feature], color="b")
            sns.distplot(reconstructed_0.iloc[selected_data.index][feature], color="g")
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
