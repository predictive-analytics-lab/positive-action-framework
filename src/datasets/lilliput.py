"""The hand crafted synthetic data."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from ethicml import Dataset, DataTuple
from scipy import stats


def lilliput(*, seed, num_samples, alpha):
    """Make the `lilliput` dataset."""
    num_gen = np.random.default_rng(seed)

    # Green = 1, Blue = 0
    s = num_gen.binomial(1, alpha, num_samples)
    cf_s = np.ones_like(s) - s

    potions_score_nrm = num_gen.normal(0.65, 0.15, len(s)).round(2).clip(0, 0.95)
    potions_score_skw = stats.skewnorm(4, 0.35, 0.2).rvs(len(s)).round(2).clip(0, 0.95)

    potions_score = np.where(s == 0, potions_score_skw, potions_score_nrm)
    cf_potions_score = np.where(s == 1, potions_score_skw, potions_score_nrm)

    pot_bane_err = num_gen.normal(0.03, 0.02, len(s)).clip(0, 1)
    potions_bane = (potions_score + pot_bane_err * (1 - s)).round(2).clip(0, 1)
    cf_potions_bane = (cf_potions_score + pot_bane_err * (1 - cf_s)).round(2).clip(0, 1)

    pot_wolf_err = num_gen.normal(0.01, 0.04, len(s)).clip(0, 1)
    potions_wolf = (potions_score + pot_wolf_err * s).round(2).clip(0, 1)
    cf_potions_wolf = (cf_potions_score + pot_wolf_err * cf_s).round(2).clip(0, 1)

    # potions_bane = [
    #     (b + a * (1 - c)).round(2).clip(0, 1)
    #     for (a, b, c) in zip(num_gen.normal(0.03, 0.02, len(s)).clip(0, 1), potions_score, s)
    # ]
    # potions_wolf = [
    #     (b + a * c).round(2).clip(0, 1)
    #     for (a, b, c) in zip(num_gen.normal(0.01, 0.04, len(s)).clip(0, 1), potions_score, s)
    # ]

    video_mean = 0.4 + 0.01 * -(s * 2 - 1)
    cf_video_mean = 0.4 + 0.01 * -(cf_s * 2 - 1)

    vid_score_nrm = num_gen.normal(0, 0.2, len(s))
    video_score = video_mean + vid_score_nrm
    cf_video_score = cf_video_mean + vid_score_nrm

    # video_score = num_gen.normal(0.4 + 0.01 * -(s * 2 - 1), 0.2).round(2).clip(0, 1)
    vid_bane_err = num_gen.normal(0, 0.02, len(s)).clip(0, 1)
    video_bane = (video_score + vid_bane_err).round(2).clip(0, 1)
    cf_video_bane = (cf_video_score + vid_bane_err).round(2).clip(0, 1)

    vid_wolf_err = num_gen.normal(0, 0.05, len(s)).clip(0, 1)
    video_wolf = (video_score + vid_wolf_err).round(2).clip(0, 1)
    cf_video_wolf = (cf_video_score + vid_wolf_err).round(2).clip(0, 1)

    # video_bane = [
    #     (v + r).round(2).clip(0, 1) for (r, v) in zip(num_gen.normal(0, 0.02, len(s)).clip(0, 1), video_score)
    # ]
    # video_wolf = [
    #     (v + r).round(2).clip(0, 1) for (r, v) in zip(num_gen.normal(0, 0.05, len(s)).clip(0, 1), video_score)
    # ]

    essay_score_vnm = num_gen.vonmises(0.4, 60, len(s)).round(2).clip(0, 1)
    essay_score_lap = num_gen.laplace(0.5, 0.075, len(s)).round(2).clip(0, 1)

    essay_score = np.where(s == 1, essay_score_lap, essay_score_vnm)
    cf_essay_score = np.where(s == 0, essay_score_lap, essay_score_vnm)

    ess_bane_err = num_gen.normal(0.03, 0.02, len(s)).clip(0, 1)
    essay_bane = (essay_score + ess_bane_err).round(2).clip(0, 1)
    cf_essay_bane = (cf_essay_score + ess_bane_err).round(2).clip(0, 1)

    ess_wolf_err = num_gen.normal(0.01, 0.01, len(s)).clip(0, 1)
    essay_wolf = (essay_score + ess_wolf_err).round(2).clip(0, 1)
    cf_essay_wolf = (cf_essay_score + ess_wolf_err).round(2).clip(0, 1)

    # essay_score = []
    # for _s in s:
    #     if _s == 0:
    #         essay_score.append(num_gen.vonmises(0.4, 60, 1).round(2).clip(0, 1))
    #     else:
    #         essay_score.append(num_gen.laplace(0.5, 0.075, 1).round(2).clip(0, 1))

    # essay_bane = [
    #     (b + a).round(2).clip(0, 1) for (a, b, c) in zip(num_gen.normal(0.03, 0.02, len(s)).clip(0, 1), essay_score, s)
    # ]
    # essay_wolf = [
    #     (b + a).round(2).clip(0, 1) for (a, b, c) in zip(num_gen.normal(0.01, 0.01, len(s)).clip(0, 1), essay_score, s)
    # ]

    potions_score = pd.DataFrame(potions_score, columns=["potions_score"])
    potions_bane = pd.DataFrame(potions_bane, columns=["potions_bane"])
    potions_wolf = pd.DataFrame(potions_wolf, columns=["potions_wolf"])

    cf_potions_score = pd.DataFrame(cf_potions_score, columns=["potions_score"])
    cf_potions_bane = pd.DataFrame(cf_potions_bane, columns=["potions_bane"])
    cf_potions_wolf = pd.DataFrame(cf_potions_wolf, columns=["potions_wolf"])

    video_score = pd.DataFrame(video_score, columns=["video_score"])
    video_bane = pd.DataFrame(video_bane, columns=["video_bane"])
    video_wolf = pd.DataFrame(video_wolf, columns=["video_wolf"])

    cf_video_score = pd.DataFrame(cf_video_score, columns=["video_score"])
    cf_video_bane = pd.DataFrame(cf_video_bane, columns=["video_bane"])
    cf_video_wolf = pd.DataFrame(cf_video_wolf, columns=["video_wolf"])

    essay_score = pd.DataFrame(essay_score, columns=["essay_score"])
    essay_bane = pd.DataFrame(essay_bane, columns=["essay_bane"])
    essay_wolf = pd.DataFrame(essay_wolf, columns=["essay_wolf"])

    cf_essay_score = pd.DataFrame(cf_essay_score, columns=["essay_score"])
    cf_essay_bane = pd.DataFrame(cf_essay_bane, columns=["essay_bane"])
    cf_essay_wolf = pd.DataFrame(cf_essay_wolf, columns=["essay_wolf"])

    s = pd.DataFrame(s, columns=["sens"])
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
    ).round(2)

    cf_data["admittance_score"] = (
        0.4 * ((cf_data["potions_bane"] + cf_data["potions_wolf"]) / 2)
        + 0.4 * ((cf_data["video_bane"] + cf_data["video_wolf"]) / 2)
        + 0.2 * ((cf_data["essay_bane"] + cf_data["essay_wolf"]) / 2)
    ).round(2)

    graduation = []
    for (c, p, v, e) in zip(data["sens"], data["potions_score"], data["video_score"], data["essay_score"]):
        if c == 0:
            graduation.append(round(0.7 * p + 0.15 * v + 0.15 * e, 2))
        else:
            graduation.append(round(0.1 * p + 0.7 * v + 0.2 * e, 2))

    cf_graduation = []
    for (c, p, v, e) in zip(cf_data["sens"], cf_data["potions_score"], cf_data["video_score"], cf_data["essay_score"]):
        if c == 0:
            cf_graduation.append(round(0.7 * p + 0.15 * v + 0.15 * e, 2))
        else:
            cf_graduation.append(round(0.1 * p + 0.7 * v + 0.2 * e, 2))

    g = pd.DataFrame(graduation, columns=["graduation_grade"])
    cf_g = pd.DataFrame(cf_graduation, columns=["graduation_grade"])
    data = pd.concat([data, g], axis=1)
    cf_data = pd.concat([cf_data, cf_g], axis=1)

    passed_initial_screening = data.nlargest(n=int(data.shape[0] * 0.2), columns='admittance_score')
    cf_passed_initial_screening = cf_data.nlargest(n=int(cf_data.shape[0] * 0.2), columns='admittance_score')

    data["accepted"] = (data.where(passed_initial_screening.isin(data), 0)["admittance_score"] > 0).astype(int)
    cf_data["accepted"] = (cf_data.where(cf_passed_initial_screening.isin(cf_data), 0)["admittance_score"] > 0).astype(
        int
    )

    data["graduation_grade>70%"] = (data["graduation_grade"] >= 0.70).astype(int)
    cf_data["graduation_grade>70%"] = (cf_data["graduation_grade"] >= 0.70).astype(int)

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
    class_prefix = ["accepted", "graduation", "admittance"]

    sns.distplot(data[(data['sens'] == 1)]["potions_wolf"], color='g')
    sns.distplot(data[(data['sens'] == 0)]['potions_wolf'], color='b')
    plt.savefig((Path(__file__).parent / "potions_wolf.png"))
    plt.clf()
    sns.distplot(cf_data[(data['sens'] == 1)]["potions_wolf"], color='g')
    sns.distplot(cf_data[(data['sens'] == 0)]['potions_wolf'], color='b')
    plt.savefig((Path(__file__).parent / "cf_potions_wolf.png"))
    plt.clf()
    sns.distplot(data[(data['sens'] == 1)]["potions_bane"], color='g')
    sns.distplot(data[(data['sens'] == 0)]['potions_bane'], color='b')
    plt.savefig((Path(__file__).parent / "potions_bane.png"))
    plt.clf()
    sns.distplot(cf_data[(data['sens'] == 1)]["potions_bane"], color='g')
    sns.distplot(cf_data[(data['sens'] == 0)]['potions_bane'], color='b')
    plt.savefig((Path(__file__).parent / "cfpotions_bane.png"))
    plt.clf()

    sns.distplot(data[(data['sens'] == 1)]["graduation_grade"], color='g')
    sns.distplot(data[(data['sens'] == 0)]['graduation_grade'], color='b')
    plt.savefig((Path(__file__).parent / "grad_grade.png"))
    plt.clf()
    sns.histplot(data=data, x="accepted", hue="sens", multiple="dodge")
    plt.savefig((Path(__file__).parent / "accepted.png"))
    plt.clf()
    sns.distplot(data[(data['sens'] == 1)]["admittance_score"], color='g')
    sns.distplot(data[(data['sens'] == 0)]['admittance_score'], color='b')
    plt.savefig((Path(__file__).parent / "admittance.png"))
    plt.clf()

    sns.distplot(cf_data[(data['sens'] == 1)]["graduation_grade"], color='g')
    sns.distplot(cf_data[(data['sens'] == 0)]['graduation_grade'], color='b')
    plt.savefig((Path(__file__).parent / "cf_grad_grade.png"))
    plt.clf()
    sns.histplot(data=cf_data, x="accepted", hue="sens", multiple="dodge")
    plt.savefig((Path(__file__).parent / "cf_accepted.png"))
    plt.clf()
    sns.distplot(cf_data[(data['sens'] == 1)]["admittance_score"], color='g')
    sns.distplot(cf_data[(data['sens'] == 0)]['admittance_score'], color='b')
    plt.savefig((Path(__file__).parent / "cf_admittance.png"))
    plt.clf()

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

    return (
        dataset,
        DataTuple(
            x=data[dataset.discrete_features + dataset.continuous_features],
            s=data[dataset.sens_attrs],
            y=data[dataset.class_labels],
        ),
        DataTuple(
            x=cf_data[dataset.discrete_features + dataset.continuous_features],
            s=cf_data[dataset.sens_attrs],
            y=cf_data[dataset.class_labels],
        ),
        DataTuple(
            x=data[dataset.discrete_features + dataset.continuous_features],
            s=data[dataset.sens_attrs],
            y=data[["graduation_grade>70%"]],
        ),
    )
