"""Test the selection rules."""

import numpy as np
import pandas as pd
import torch
from ethicml import Prediction

from paf.selection import produce_selection_groups, selection_rules
from paf.utils import facct_mapper, facct_mapper_2, facct_mapper_outcomes


def test_selection_rule_first_4():
    """Test the selection rules."""
    all_s0_s0_preds = torch.zeros(10, 1, dtype=int)
    all_s0_s1_preds = torch.zeros(10, 1, dtype=int)
    all_s1_s0_preds = torch.zeros(10, 1, dtype=int)
    all_s1_s1_preds = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1, 1, 1]).unsqueeze(1)

    all_s = torch.tensor([0, 1, 1, 0, 0, 0, 1, 1, 1, 1]).unsqueeze(1)
    all_p = torch.tensor([1, 0, 0, 1, 1, 1, 1, 1, 1, 1]).unsqueeze(1)

    pd_results = pd.DataFrame(
        torch.cat(
            [all_s0_s0_preds, all_s0_s1_preds, all_s1_s0_preds, all_s1_s1_preds, all_s, all_p],
            dim=1,
        )
        .cpu()
        .numpy(),
        columns=["s1_0_s2_0", "s1_0_s2_1", "s1_1_s2_0", "s1_1_s2_1", "true_s", "actual"],
    )

    decision = selection_rules(pd_results)

    np.testing.assert_array_equal(decision, np.array([0, 1, 1, 2, 2, 2, 3, 3, 3, 3]))

    to_return = facct_mapper(Prediction(hard=pd.Series(decision)))

    pd.testing.assert_series_equal(to_return.hard, pd.Series([5, 6, 6, 3, 3, 3, 4, 4, 4, 4]))

    next_up = facct_mapper_outcomes(facct_mapper_2(to_return), fair=False)

    pd.testing.assert_series_equal(next_up.hard, pd.Series([0, 0, 0, 0, 0, 0, 1, 1, 1, 1]))

    preds = produce_selection_groups(pd_results, None)

    pd.testing.assert_series_equal(preds.hard, pd.Series([0, 0, 0, 0, 0, 0, 1, 1, 1, 1]))


def test_selection_rule_four_to_eight():
    """Test the selection rules."""
    all_s0_s0_preds = torch.zeros(10, 1, dtype=int)
    all_s0_s1_preds = torch.zeros(10, 1, dtype=int)
    all_s1_s0_preds = torch.ones(10, 1, dtype=int)
    all_s1_s1_preds = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1, 1, 1]).unsqueeze(1)

    all_s = torch.tensor([0, 1, 1, 0, 0, 0, 1, 1, 1, 1]).unsqueeze(1)
    all_p = torch.tensor([1, 0, 0, 1, 1, 1, 1, 1, 1, 1]).unsqueeze(1)

    pd_results = pd.DataFrame(
        torch.cat(
            [all_s0_s0_preds, all_s0_s1_preds, all_s1_s0_preds, all_s1_s1_preds, all_s, all_p],
            dim=1,
        )
        .cpu()
        .numpy(),
        columns=["s1_0_s2_0", "s1_0_s2_1", "s1_1_s2_0", "s1_1_s2_1", "true_s", "actual"],
    )

    decision = selection_rules(pd_results)

    np.testing.assert_array_equal(decision, np.array([4, 5, 5, 6, 6, 6, 7, 7, 7, 7]))

    to_return = facct_mapper(Prediction(hard=pd.Series(decision)))

    pd.testing.assert_series_equal(to_return.hard, pd.Series([7, 7, 7, 3, 3, 3, 4, 4, 4, 4]))

    next_up = facct_mapper_outcomes(facct_mapper_2(to_return), fair=False)

    pd.testing.assert_series_equal(next_up.hard, pd.Series([0, 0, 0, 0, 0, 0, 1, 1, 1, 1]))

    preds = produce_selection_groups(pd_results, None)

    pd.testing.assert_series_equal(preds.hard, pd.Series([0, 0, 0, 0, 0, 0, 1, 1, 1, 1]))


def test_selection_rule_eight_to_fourteen():
    """Test the selection rules."""
    all_s0_s0_preds = torch.zeros(36, 1, dtype=int)

    all_s0_s1_preds = torch.ones(36, 1, dtype=int)

    all_s1_s0_preds = torch.cat(
        (torch.zeros(10, 1, dtype=int), torch.ones(26, 1, dtype=int)), dim=0
    )

    all_s1_s1_preds = torch.tensor(
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        + [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ).unsqueeze(1)

    all_s = torch.tensor(
        [0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1]
        + [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    ).unsqueeze(1)
    all_p = torch.tensor(
        [1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
        + [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ).unsqueeze(1)

    pd_results = pd.DataFrame(
        torch.cat(
            [
                all_s0_s0_preds,
                all_s0_s1_preds,
                all_s1_s0_preds,
                all_s1_s1_preds,
                all_s,
                all_p,
            ],
            dim=1,
        )
        .cpu()
        .numpy(),
        columns=["s1_0_s2_0", "s1_0_s2_1", "s1_1_s2_0", "s1_1_s2_1", "true_s", "actual"],
    )

    decision = selection_rules(pd_results)

    np.testing.assert_array_equal(
        decision,
        np.array(
            [8, 9, 9, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13]
            + [13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15]
        ),
    )

    # print(decision)

    to_return = facct_mapper(Prediction(hard=pd.Series(decision)))

    pd.testing.assert_series_equal(
        to_return.hard,
        pd.Series(
            [7, 7, 7, 1, 1, 1, 4, 4, 4, 4, 7, 7, 7, 7, 7, 7, 7, 7]
            + [7, 7, 7, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2]
        ),
    )

    next_up = facct_mapper_2(to_return)

    pd.testing.assert_series_equal(
        next_up.hard,
        pd.Series(
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
            + [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        ),
    )

    preds = produce_selection_groups(pd_results, None)

    pd.testing.assert_series_equal(
        preds.hard,
        pd.Series(
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
            + [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        ),
    )


def test_selection_rule_fourteen_to_twenty_two():
    """Test the selection rules."""
    all_s0_s0_preds = torch.cat((torch.zeros(3, 1, dtype=int), torch.ones(33, 1, dtype=int)), dim=0)
    all_s0_s1_preds = torch.cat((torch.ones(3, 1, dtype=int), torch.zeros(33, 1, dtype=int)), dim=0)
    all_s1_s0_preds = torch.tensor(
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        + [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ).unsqueeze(1)

    all_s1_s1_preds = torch.tensor(
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        + [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ).unsqueeze(1)

    all_s = torch.tensor(
        [0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1]
        + [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    ).unsqueeze(1)
    all_p = torch.tensor(
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        + [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    ).unsqueeze(1)

    pd_results = pd.DataFrame(
        torch.cat(
            [
                all_s0_s0_preds,
                all_s0_s1_preds,
                all_s1_s0_preds,
                all_s1_s1_preds,
                all_s,
                all_p,
            ],
            dim=1,
        )
        .cpu()
        .numpy(),
        columns=["s1_0_s2_0", "s1_0_s2_1", "s1_1_s2_0", "s1_1_s2_1", "true_s", "actual"],
    )

    decision = selection_rules(pd_results)

    np.testing.assert_array_equal(
        decision,
        np.array(
            [14, 15, 15, 16, 16, 16, 17, 17, 17, 17, 18, 18, 18, 18, 18, 19, 19, 19]
            + [19, 19, 19, 20, 20, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 21, 21, 21]
        ),
    )

    # print(decision)

    to_return = facct_mapper(Prediction(hard=pd.Series(decision)))

    pd.testing.assert_series_equal(
        to_return.hard,
        pd.Series(
            [1, 2, 2, 8, 8, 8, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8]
            + [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7, 7, 7, 7, 7, 7, 7, 7]
        ),
    )

    next_up = facct_mapper_outcomes(facct_mapper_2(to_return), fair=False)

    pd.testing.assert_series_equal(
        next_up.hard,
        pd.Series(
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ),
    )

    preds = produce_selection_groups(pd_results, None)

    pd.testing.assert_series_equal(
        preds.hard,
        pd.Series(
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ),
    )


def test_selection_rule_twenty_two_to_thirty_two():
    """Test the selection rules."""
    all_s0_s0_preds = torch.ones(55, 1, dtype=int)
    all_s0_s1_preds = torch.cat((torch.zeros(3, 1, dtype=int), torch.ones(52, 1, dtype=int)), dim=0)
    all_s1_s0_preds = torch.tensor(
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
        + [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ).unsqueeze(1)
    all_s1_s1_preds = torch.tensor(
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
        + [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ).unsqueeze(1)

    all_s = torch.tensor(
        [0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1]
        + [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ).unsqueeze(1)
    all_p = torch.tensor(
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
        + [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ).unsqueeze(1)

    pd_results = pd.DataFrame(
        torch.cat(
            [
                all_s0_s0_preds,
                all_s0_s1_preds,
                all_s1_s0_preds,
                all_s1_s1_preds,
                all_s,
                all_p,
            ],
            dim=1,
        )
        .cpu()
        .numpy(),
        columns=["s1_0_s2_0", "s1_0_s2_1", "s1_1_s2_0", "s1_1_s2_1", "true_s", "actual"],
    )

    decision = selection_rules(pd_results)

    np.testing.assert_array_equal(
        decision,
        np.array(
            [22, 23, 23, 24, 24, 24, 25, 25, 25, 25, 26, 26, 26, 26, 26, 27, 27, 27, 27, 27, 27]
            + [28, 28, 28, 28, 28, 28, 28, 29, 29, 29, 29, 29, 29, 29, 29, 30, 30]
            + [30, 30, 30, 30, 30, 30, 30, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31]
        ),
    )

    to_return = facct_mapper(Prediction(hard=pd.Series(decision)))

    pd.testing.assert_series_equal(
        to_return.hard,
        pd.Series(
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
            + [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
            + [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        ),
    )

    next_up = facct_mapper_2(to_return)

    pd.testing.assert_series_equal(
        next_up.hard,
        pd.Series(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
            + [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        ),
    )

    preds = produce_selection_groups(pd_results, None)

    pd.testing.assert_series_equal(
        preds.hard,
        pd.Series(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            + [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        ),
    )


def test_selection_rule():
    """Test the selection rules."""
    all_s0_s0_preds = torch.cat(
        (torch.zeros(16, 1, dtype=int), torch.ones(16, 1, dtype=int)), dim=0
    )

    all_s0_s1_preds = torch.cat(
        (
            torch.zeros(8, 1, dtype=int),
            torch.ones(8, 1, dtype=int),
            torch.zeros(8, 1, dtype=int),
            torch.ones(8, 1, dtype=int),
        ),
        dim=0,
    )
    all_s1_s0_preds = torch.cat(
        (
            torch.zeros(4, 1, dtype=int),
            torch.ones(4, 1, dtype=int),
            torch.zeros(4, 1, dtype=int),
            torch.ones(4, 1, dtype=int),
            torch.zeros(4, 1, dtype=int),
            torch.ones(4, 1, dtype=int),
            torch.zeros(4, 1, dtype=int),
            torch.ones(4, 1, dtype=int),
        ),
        dim=0,
    )
    all_s1_s1_preds = torch.tensor(
        [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0]
        + [1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
    ).unsqueeze(1)

    all_s = torch.tensor(
        [0, 1, 0, 1, 0, 1, 0, 1, 0]
        + [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    ).unsqueeze(1)
    all_p = torch.tensor(
        [1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1]
        + [1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1]
    ).unsqueeze(1)

    pd_results = pd.DataFrame(
        torch.cat(
            [
                all_s0_s0_preds,
                all_s0_s1_preds,
                all_s1_s0_preds,
                all_s1_s1_preds,
                all_s,
                all_p,
            ],
            dim=1,
        )
        .cpu()
        .numpy(),
        columns=["s1_0_s2_0", "s1_0_s2_1", "s1_1_s2_0", "s1_1_s2_1", "true_s", "actual"],
    )

    decision = selection_rules(pd_results)

    np.testing.assert_array_equal(
        decision,
        np.array(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
            + [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
        ),
    )

    # print(decision)

    to_return = facct_mapper(Prediction(hard=pd.Series(decision)))

    pd.testing.assert_series_equal(
        to_return.hard,
        pd.Series(
            [5, 6, 3, 4, -1, -1, 3, 4, -1, -1, 1, 4, -1, -1, 1, 2]
            + [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 2]
        ),
    )

    next_up = facct_mapper_2(to_return)

    pd.testing.assert_series_equal(
        next_up.hard,
        pd.Series(
            [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0]
            + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
        ),
    )

    preds = produce_selection_groups(pd_results, None)

    pd.testing.assert_series_equal(
        preds.hard,
        pd.Series(
            [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0]
            + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
        ),
    )
