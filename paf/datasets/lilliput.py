"""The hand crafted synthetic data."""
from __future__ import annotations

from typing import NamedTuple

from ethicml import Dataset, DataTuple, Prediction
import numpy as np
import pandas as pd
import scipy
from scipy import stats

from paf.selection import produce_selection_groups

import numpy.typing as npt


class CfData(NamedTuple):
    dataset: Dataset
    data: DataTuple
    cf_data: DataTuple
    data_true_outcome: DataTuple
    cf_groups: Prediction
    data_xs0_ys0: DataTuple
    data_xs0_ys1: DataTuple
    data_xs1_ys0: DataTuple
    data_xs1_ys1: DataTuple


def lilliput(*, seed: int, num_samples: int, alpha: float, gamma: float) -> CfData:
    """Make the `lilliput` dataset."""
    num_gen = np.random.default_rng(seed)

    # Green is 1, Blue is 0
    s = num_gen.binomial(1, alpha, num_samples)
    cf_s: npt.NDArray[np.int_] = np.ones_like(s) - s
    s_all_0 = np.zeros_like(s)
    s_all_1 = np.ones_like(s)

    potions_percent = num_gen.random(len(s))

    potions_score_nrm = (
        scipy.stats.norm.ppf(potions_percent, loc=0.65, scale=0.15).round(2).clip(0, 1)
    )
    potions_score_skw = (
        stats.johnsonsu.ppf(potions_percent, a=-2, b=3, loc=0.35, scale=0.2).round(2).clip(0, 1)
    )

    potions_score = np.where(s == 0, potions_score_skw, potions_score_nrm)
    cf_potions_score = np.where(s == 1, potions_score_skw, potions_score_nrm)
    potions_score_all_0 = potions_score_skw
    potions_score_all_1 = potions_score_nrm

    pot_bane_err = num_gen.normal(0.03, 0.02, len(s)).clip(0, 1)
    potions_bane = (potions_score + pot_bane_err * ((2 * s) - 1)).round(2).clip(0, 1)
    cf_potions_bane = (cf_potions_score + pot_bane_err * ((2 * cf_s) - 1)).round(2).clip(0, 1)
    potions_bane_all_0 = (
        (potions_score_all_0 + pot_bane_err * ((2 * s_all_0) - 1)).round(2).clip(0, 1)
    )
    potions_bane_all_1 = (
        (potions_score_all_1 + pot_bane_err * ((2 * s_all_1) - 1)).round(2).clip(0, 1)
    )

    pot_wolf_err = num_gen.normal(0.01, 0.04, len(s)).clip(0, 1)
    potions_wolf = (potions_score + pot_wolf_err * ((2 * s) - 1)).round(2).clip(0, 1)
    cf_potions_wolf = (cf_potions_score + pot_wolf_err * ((2 * cf_s) - 1)).round(2).clip(0, 1)
    potions_wolf_all_0 = (
        (potions_score_all_0 + pot_wolf_err * ((2 * s_all_0) - 1)).round(2).clip(0, 1)
    )
    potions_wolf_all_1 = (
        (potions_score_all_1 + pot_wolf_err * ((2 * s_all_1) - 1)).round(2).clip(0, 1)
    )

    video_mean = 0.4 + 0.01 * (s * 2 - 1)
    cf_video_mean = 0.4 + 0.01 * (cf_s * 2 - 1)
    video_mean_all_0 = 0.4 + 0.01 * (s_all_0 * 2 - 1)
    video_mean_all_1 = 0.4 + 0.01 * (s_all_1 * 2 - 1)

    vid_score_nrm = num_gen.normal(0, 0.2, len(s))
    video_score: np.ndarray = video_mean + vid_score_nrm
    cf_video_score: np.ndarray = cf_video_mean + vid_score_nrm
    video_score_all_0: np.ndarray = video_mean_all_0 + vid_score_nrm
    video_score_all_1: np.ndarray = video_mean_all_1 + vid_score_nrm

    vid_bane_err = num_gen.normal(0, 0.02, len(s)).clip(0, 1)
    video_bane = (video_score + vid_bane_err).round(2).clip(0, 1)
    cf_video_bane = (cf_video_score + vid_bane_err).round(2).clip(0, 1)
    video_bane_all_0 = (video_score_all_0 + vid_bane_err).round(2).clip(0, 1)
    video_bane_all_1 = (video_score_all_1 + vid_bane_err).round(2).clip(0, 1)

    vid_wolf_err = num_gen.normal(0, 0.05, len(s)).clip(0, 1)
    video_wolf = (video_score + vid_wolf_err).round(2).clip(0, 1)
    cf_video_wolf = (cf_video_score + vid_wolf_err).round(2).clip(0, 1)
    video_wolf_all_0 = (video_score_all_0 + vid_wolf_err).round(2).clip(0, 1)
    video_wolf_all_1 = (video_score_all_1 + vid_wolf_err).round(2).clip(0, 1)

    random_nums = num_gen.random(len(s))

    essay_score_vnm = (
        scipy.stats.t.ppf(random_nums, df=100, loc=0.4, scale=0.15).round(2).clip(0, 1)
    )
    essay_score_lap = scipy.stats.laplace.ppf(random_nums, loc=0.5, scale=0.075).round(2).clip(0, 1)

    essay_score = np.where(s == 1, essay_score_lap, essay_score_vnm)
    cf_essay_score = np.where(s == 0, essay_score_lap, essay_score_vnm)
    essay_score_all_0 = essay_score_lap
    essay_score_all_1 = essay_score_vnm

    ess_bane_err = num_gen.normal(0.03, 0.01, len(s)).clip(0, 1)
    essay_bane = (essay_score + ess_bane_err).round(2).clip(0, 1)
    cf_essay_bane = (cf_essay_score + ess_bane_err).round(2).clip(0, 1)
    essay_bane_all_0 = (essay_score_all_0 + ess_bane_err).round(2).clip(0, 1)
    essay_bane_all_1 = (essay_score_all_1 + ess_bane_err).round(2).clip(0, 1)

    ess_wolf_err = num_gen.normal(0.01, 0.02, len(s)).clip(0, 1)
    essay_wolf = (essay_score + ess_wolf_err).round(2).clip(0, 1)
    cf_essay_wolf = (cf_essay_score + ess_wolf_err).round(2).clip(0, 1)
    essay_wolf_all_0 = (essay_score_all_0 + ess_wolf_err).round(2).clip(0, 1)
    essay_wolf_all_1 = (essay_score_all_1 + ess_wolf_err).round(2).clip(0, 1)

    potions_score = pd.DataFrame(potions_score, columns=["potions_score"])
    potions_bane = pd.DataFrame(potions_bane, columns=["potions_bane"])
    potions_wolf = pd.DataFrame(potions_wolf, columns=["potions_wolf"])

    potions_score_all_0 = pd.DataFrame(potions_score_all_0, columns=["potions_score"])
    potions_bane_all_0 = pd.DataFrame(potions_bane_all_0, columns=["potions_bane"])
    potions_wolf_all_0 = pd.DataFrame(potions_wolf_all_0, columns=["potions_wolf"])

    potions_score_all_1 = pd.DataFrame(potions_score_all_1, columns=["potions_score"])
    potions_bane_all_1 = pd.DataFrame(potions_bane_all_1, columns=["potions_bane"])
    potions_wolf_all_1 = pd.DataFrame(potions_wolf_all_1, columns=["potions_wolf"])

    cf_potions_score = pd.DataFrame(cf_potions_score, columns=["potions_score"])
    cf_potions_bane = pd.DataFrame(cf_potions_bane, columns=["potions_bane"])
    cf_potions_wolf = pd.DataFrame(cf_potions_wolf, columns=["potions_wolf"])

    video_score = pd.DataFrame(video_score, columns=["video_score"])
    video_bane = pd.DataFrame(video_bane, columns=["video_bane"])
    video_wolf = pd.DataFrame(video_wolf, columns=["video_wolf"])

    video_score_all_0 = pd.DataFrame(video_score_all_0, columns=["video_score"])
    video_bane_all_0 = pd.DataFrame(video_bane_all_0, columns=["video_bane"])
    video_wolf_all_0 = pd.DataFrame(video_wolf_all_0, columns=["video_wolf"])

    video_score_all_1 = pd.DataFrame(video_score_all_1, columns=["video_score"])
    video_bane_all_1 = pd.DataFrame(video_bane_all_1, columns=["video_bane"])
    video_wolf_all_1 = pd.DataFrame(video_wolf_all_1, columns=["video_wolf"])

    cf_video_score = pd.DataFrame(cf_video_score, columns=["video_score"])
    cf_video_bane = pd.DataFrame(cf_video_bane, columns=["video_bane"])
    cf_video_wolf = pd.DataFrame(cf_video_wolf, columns=["video_wolf"])

    essay_score = pd.DataFrame(essay_score, columns=["essay_score"])
    essay_bane = pd.DataFrame(essay_bane, columns=["essay_bane"])
    essay_wolf = pd.DataFrame(essay_wolf, columns=["essay_wolf"])

    essay_score_all_0 = pd.DataFrame(essay_score_all_0, columns=["essay_score"])
    essay_bane_all_0 = pd.DataFrame(essay_bane_all_0, columns=["essay_bane"])
    essay_wolf_all_0 = pd.DataFrame(essay_wolf_all_0, columns=["essay_wolf"])

    essay_score_all_1 = pd.DataFrame(essay_score_all_1, columns=["essay_score"])
    essay_bane_all_1 = pd.DataFrame(essay_bane_all_1, columns=["essay_bane"])
    essay_wolf_all_1 = pd.DataFrame(essay_wolf_all_1, columns=["essay_wolf"])

    cf_essay_score = pd.DataFrame(cf_essay_score, columns=["essay_score"])
    cf_essay_bane = pd.DataFrame(cf_essay_bane, columns=["essay_bane"])
    cf_essay_wolf = pd.DataFrame(cf_essay_wolf, columns=["essay_wolf"])

    s = pd.DataFrame(s, columns=["sens"])
    s_all_0 = pd.DataFrame(s_all_0, columns=["sens"])
    s_all_1 = pd.DataFrame(s_all_1, columns=["sens"])
    cf_s = pd.DataFrame(cf_s, columns=["sens"])
    data = pd.concat(
        [
            potions_score,
            potions_bane,
            potions_wolf,
            s,
            video_score,
            video_bane,
            video_wolf,
            essay_score,
            essay_bane,
            essay_wolf,
        ],
        axis=1,
    )

    data_all_0 = pd.concat(
        [
            potions_score_all_0,
            potions_bane_all_0,
            potions_wolf_all_0,
            s_all_0,
            video_score_all_0,
            video_bane_all_0,
            video_wolf_all_0,
            essay_score_all_0,
            essay_bane_all_0,
            essay_wolf_all_0,
        ],
        axis=1,
    )

    data_all_1 = pd.concat(
        [
            potions_score_all_1,
            potions_bane_all_1,
            potions_wolf_all_1,
            s_all_1,
            video_score_all_1,
            video_bane_all_1,
            video_wolf_all_1,
            essay_score_all_1,
            essay_bane_all_1,
            essay_wolf_all_1,
        ],
        axis=1,
    )

    cf_data = pd.concat(
        [
            cf_potions_score,
            cf_potions_bane,
            cf_potions_wolf,
            cf_s,
            cf_video_score,
            cf_video_bane,
            cf_video_wolf,
            cf_essay_score,
            cf_essay_bane,
            cf_essay_wolf,
        ],
        axis=1,
    )

    data["admittance_score"] = (
        0.4 * ((data["potions_bane"] + data["potions_wolf"]) / 2)
        + 0.4 * ((data["video_bane"] + data["video_wolf"]) / 2)
        + 0.2 * ((data["essay_bane"] + data["essay_wolf"]) / 2)
        + gamma * ((2 * data["sens"]) - 1)
    ).round(2)

    cf_data["admittance_score"] = (
        0.4 * ((cf_data["potions_bane"] + cf_data["potions_wolf"]) / 2)
        + 0.4 * ((cf_data["video_bane"] + cf_data["video_wolf"]) / 2)
        + 0.2 * ((cf_data["essay_bane"] + cf_data["essay_wolf"]) / 2)
        + gamma * ((2 * cf_data["sens"]) - 1)
    ).round(2)
    SY0_AD_SCORE = "Sy=0_admittance_score"
    SY1_AD_SCORE = "Sy=1_admittance_score"
    data_all_0[SY0_AD_SCORE] = (
        0.4 * ((data_all_0["potions_bane"] + data_all_0["potions_wolf"]) / 2)
        + 0.4 * ((data_all_0["video_bane"] + data_all_0["video_wolf"]) / 2)
        + 0.2 * ((data_all_0["essay_bane"] + data_all_0["essay_wolf"]) / 2)
        + gamma * ((2 * data_all_0["sens"]) - 1)
    ).round(2)

    data_all_0[SY1_AD_SCORE] = (
        0.4 * ((data_all_0["potions_bane"] + data_all_0["potions_wolf"]) / 2)
        + 0.4 * ((data_all_0["video_bane"] + data_all_0["video_wolf"]) / 2)
        + 0.2 * ((data_all_0["essay_bane"] + data_all_0["essay_wolf"]) / 2)
        + gamma * ((2 * data_all_1["sens"]) - 1)
    ).round(2)

    data_all_1[SY0_AD_SCORE] = (
        0.4 * ((data_all_1["potions_bane"] + data_all_1["potions_wolf"]) / 2)
        + 0.4 * ((data_all_1["video_bane"] + data_all_1["video_wolf"]) / 2)
        + 0.2 * ((data_all_1["essay_bane"] + data_all_1["essay_wolf"]) / 2)
        + gamma * ((2 * data_all_0["sens"]) - 1)
    ).round(2)

    data_all_1[SY1_AD_SCORE] = (
        0.4 * ((data_all_1["potions_bane"] + data_all_1["potions_wolf"]) / 2)
        + 0.4 * ((data_all_1["video_bane"] + data_all_1["video_wolf"]) / 2)
        + 0.2 * ((data_all_1["essay_bane"] + data_all_1["essay_wolf"]) / 2)
        + gamma * ((2 * data_all_1["sens"]) - 1)
    ).round(2)

    graduation = []
    for (c, p, v, e) in zip(
        data["sens"], data["potions_score"], data["video_score"], data["essay_score"]
    ):
        if c == 0:
            graduation.append(round(0.3 * p + 0.25 * v + 0.45 * e, 2))
        else:
            graduation.append(round(0.1 * p + 0.7 * v + 0.2 * e, 2))
    graduation_all_0 = round(  # type: ignore[call-overload]
        0.3 * data_all_0["potions_score"]
        + 0.25 * data_all_0["video_score"]
        + 0.45 * data_all_0["essay_score"],
        2,
    )
    graduation_all_1 = round(  # type: ignore[call-overload]
        0.1 * data_all_1["potions_score"]
        + 0.7 * data_all_1["video_score"]
        + 0.2 * data_all_1["essay_score"],
        2,
    )

    cf_graduation = []
    for (c, p, v, e) in zip(
        cf_data["sens"], cf_data["potions_score"], cf_data["video_score"], cf_data["essay_score"]
    ):
        if c == 0:
            cf_graduation.append(round(0.3 * p + 0.25 * v + 0.45 * e, 2))
        else:
            cf_graduation.append(round(0.1 * p + 0.7 * v + 0.2 * e, 2))

    g = pd.DataFrame(graduation, columns=["graduation_grade"])
    g_all_0 = pd.DataFrame(graduation_all_0, columns=["graduation_grade"])
    g_all_1 = pd.DataFrame(graduation_all_1, columns=["graduation_grade"])
    cf_g = pd.DataFrame(cf_graduation, columns=["graduation_grade"])
    data = pd.concat([data, g], axis=1)
    data_all_0 = pd.concat([data_all_0, g_all_0], axis=1)
    data_all_1 = pd.concat([data_all_1, g_all_1], axis=1)
    cf_data = pd.concat([cf_data, cf_g], axis=1)

    passed_initial_screening = data.nlargest(n=int(data.shape[0] * 0.2), columns='admittance_score')
    cf_passed_initial_screening = cf_data.nlargest(
        n=int(cf_data.shape[0] * 0.2), columns='admittance_score'
    )

    passed_threshold = data.nlargest(n=int(data.shape[0] * 0.2), columns='admittance_score')[
        "admittance_score"
    ].min()

    print(f"Passed threshold is: {passed_threshold}")

    accepted_sx_0_sy_0 = data_all_0[SY0_AD_SCORE] >= passed_threshold
    accepted_sx_0_sy_1 = data_all_0[SY1_AD_SCORE] >= passed_threshold
    accepted_sx_1_sy_0 = data_all_1[SY0_AD_SCORE] >= passed_threshold
    accepted_sx_1_sy_1 = data_all_1[SY1_AD_SCORE] >= passed_threshold

    gt_results = pd.concat(
        [
            accepted_sx_0_sy_0.astype(int).rename("s1_0_s2_0"),
            accepted_sx_0_sy_1.astype(int).rename("s1_0_s2_1"),
            accepted_sx_1_sy_0.astype(int).rename("s1_1_s2_0"),
            accepted_sx_1_sy_1.astype(int).rename("s1_1_s2_1"),
            data["sens"].rename("true_s"),
        ],
        axis=1,
    )

    best_aim = produce_selection_groups(gt_results, data_name="GroundTruth")

    data["accepted"] = (
        data.where(passed_initial_screening.isin(data), 0)["admittance_score"] > 0
    ).astype(int)

    cf_data["accepted"] = (
        cf_data.where(cf_passed_initial_screening.isin(cf_data), 0)["admittance_score"] > 0
    ).astype(int)

    GRAD_MT_60 = "graduation_grade>60%"
    data[GRAD_MT_60] = (data["graduation_grade"] >= 0.60).astype(int)
    data_all_0[GRAD_MT_60] = (data_all_0["graduation_grade"] >= 0.60).astype(int)
    data_all_1[GRAD_MT_60] = (data_all_1["graduation_grade"] >= 0.60).astype(int)
    cf_data[GRAD_MT_60] = (cf_data["graduation_grade"] >= 0.60).astype(int)

    GRAD_MT_70 = "graduation_grade>70%"
    data[GRAD_MT_70] = (data["graduation_grade"] >= 0.70).astype(int)
    data_all_0[GRAD_MT_70] = (data_all_0["graduation_grade"] >= 0.70).astype(int)
    data_all_1[GRAD_MT_70] = (data_all_1["graduation_grade"] >= 0.70).astype(int)
    cf_data[GRAD_MT_70] = (cf_data["graduation_grade"] >= 0.70).astype(int)

    features = [
        "potions_bane",
        "potions_wolf",
        "video_bane",
        "video_wolf",
        "essay_bane",
        "essay_wolf",
        "sens",
        "accepted",
        "admittance_score",
        "graduation_grade",
        "graduation_grade>60%",
        "graduation_grade>70%",
    ]

    cont_features = [
        "potions_bane",
        "potions_wolf",
        "video_bane",
        "video_wolf",
        "essay_bane",
        "essay_wolf",
        "admittance_score",
        "graduation_grade",
    ]

    s_prefix = ["sens"]
    sens_attr = "sens"
    class_label = "accepted"
    class_prefix = ["accepted", "graduation", "admittance", "Sy=0", "Sy=1"]

    dataset = Dataset(
        name=f"University of Lilliput",
        num_samples=num_samples,
        filename_or_path=f"lilliput.csv",
        features=features,
        cont_features=cont_features,
        s_prefix=s_prefix,
        sens_attr_spec=sens_attr,
        class_label_spec=class_label,
        class_label_prefix=class_prefix,
        discrete_only=False,
    )

    return CfData(
        dataset=dataset,
        data=DataTuple(
            x=data[dataset.discrete_features + dataset.continuous_features],
            s=data[dataset.sens_attrs],
            y=data[dataset.class_labels],
        ),
        cf_data=DataTuple(
            x=cf_data[dataset.discrete_features + dataset.continuous_features],
            s=cf_data[dataset.sens_attrs],
            y=cf_data[dataset.class_labels],
        ),
        data_true_outcome=DataTuple(
            x=data[dataset.discrete_features + dataset.continuous_features],
            s=data[dataset.sens_attrs],
            y=data[[GRAD_MT_60]],
        ),
        cf_groups=best_aim,
        data_xs0_ys0=DataTuple(
            x=data_all_0[dataset.discrete_features + dataset.continuous_features],
            s=data_all_0[dataset.sens_attrs],
            y=gt_results["s1_0_s2_0"].to_frame(),
        ),
        data_xs0_ys1=DataTuple(
            x=data_all_0[dataset.discrete_features + dataset.continuous_features],
            s=data_all_0[dataset.sens_attrs],
            y=gt_results["s1_0_s2_1"].to_frame(),
        ),
        data_xs1_ys0=DataTuple(
            x=data_all_1[dataset.discrete_features + dataset.continuous_features],
            s=data_all_1[dataset.sens_attrs],
            y=gt_results["s1_1_s2_0"].to_frame(),
        ),
        data_xs1_ys1=DataTuple(
            x=data_all_1[dataset.discrete_features + dataset.continuous_features],
            s=data_all_1[dataset.sens_attrs],
            y=gt_results["s1_1_s2_1"].to_frame(),
        ),
    )
