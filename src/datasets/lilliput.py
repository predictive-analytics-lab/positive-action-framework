"""The hand crafted synthetic data."""
import numpy as np
import pandas as pd
from ethicml import Dataset, DataTuple
from scipy import stats


def lilliput(*, seed, num_samples, alpha):
    """Make the `lilliput` dataset."""
    num_gen = np.random.default_rng(seed)

    # Green = 1, Blue = 0
    s = num_gen.binomial(1, alpha, num_samples)

    potions_score = []
    for _s in s:
        if _s == 0:
            potions_score.append(num_gen.normal(0.65, 0.15, 1).round(2).clip(0, 0.95))
        else:
            potions_score.append(stats.skewnorm(4, 0.35, 0.2).rvs(1).round(2).clip(0, 0.95))

    potions_bane = [
        (b + a * (1 - c)).round(2).clip(0, 1)
        for (a, b, c) in zip(num_gen.normal(0.03, 0.02, len(s)).clip(0, 1), potions_score, s)
    ]
    potions_wolf = [
        (b + a * c).round(2).clip(0, 1)
        for (a, b, c) in zip(num_gen.normal(0.01, 0.04, len(s)).clip(0, 1), potions_score, s)
    ]

    video_score = num_gen.normal(0.4 + 0.01 * -(s * 2 - 1), 0.2).round(2).clip(0, 1)
    video_bane = [
        (v + r).round(2).clip(0, 1) for (r, v) in zip(num_gen.normal(0, 0.02, len(s)).clip(0, 1), video_score)
    ]
    video_wolf = [
        (v + r).round(2).clip(0, 1) for (r, v) in zip(num_gen.normal(0, 0.05, len(s)).clip(0, 1), video_score)
    ]

    essay_score = []
    for _s in s:
        if _s == 0:
            essay_score.append(num_gen.vonmises(0.4, 60, 1).round(2).clip(0, 1))
        else:
            essay_score.append(num_gen.laplace(0.5, 0.075, 1).round(2).clip(0, 1))

    essay_bane = [
        (b + a).round(2).clip(0, 1) for (a, b, c) in zip(num_gen.normal(0.03, 0.02, len(s)).clip(0, 1), essay_score, s)
    ]
    essay_wolf = [
        (b + a).round(2).clip(0, 1) for (a, b, c) in zip(num_gen.normal(0.01, 0.01, len(s)).clip(0, 1), essay_score, s)
    ]

    potions_score = pd.DataFrame(potions_score, columns=["potions_score"])
    potions_bane = pd.DataFrame(potions_bane, columns=["potions_bane"])
    potions_wolf = pd.DataFrame(potions_wolf, columns=["potions_wolf"])

    video_score = pd.DataFrame(video_score, columns=["video_score"])
    video_bane = pd.DataFrame(video_bane, columns=["video_bane"])
    video_wolf = pd.DataFrame(video_wolf, columns=["video_wolf"])

    essay_score = pd.DataFrame(essay_score, columns=["essay_score"])
    essay_bane = pd.DataFrame(essay_bane, columns=["essay_bane"])
    essay_wolf = pd.DataFrame(essay_wolf, columns=["essay_wolf"])

    s = pd.DataFrame(s, columns=["colour"])
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

    data["admittance_score"] = (
        0.4 * ((data["potions_bane"] + data["potions_wolf"]) / 2)
        + 0.4 * ((data["video_bane"] + data["video_wolf"]) / 2)
        + 0.2 * ((data["essay_bane"] + data["essay_wolf"]) / 2)
    ).round(2)

    graduation = []
    for (c, p, v, e) in zip(data["colour"], data["potions_score"], data["video_score"], data["essay_score"]):
        if c == 0:
            graduation.append(round(0.7 * p + 0.15 * v + 0.15 * e, 2))
        else:
            graduation.append(round(0.1 * p + 0.7 * v + 0.2 * e, 2))

    g = pd.DataFrame(graduation, columns=["graduation_grade"])
    data = pd.concat([data, g], axis=1)

    passed_initial_screening = data.nlargest(n=int(data.shape[0] * 0.2), columns='admittance_score')

    data["accepted"] = (data.where(passed_initial_screening.isin(data), 0)["admittance_score"] > 0).astype(int)

    features = [
        "potions_bane",
        "potions_wolf",
        "video_bane",
        "video_wolf",
        "essay_bane",
        "essay_wolf",
        "colour",
        "accepted",
        "admittance_score",
        "graduation_grade",
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

    s_prefix = ["colour"]
    sens_attr = "colour"
    class_label = "accepted"
    class_prefix = ["accepted", "graduation", "admittance"]

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
            x=data[dataset.discrete_features + dataset.cont_features],
            s=data[dataset.sens_attrs],
            y=data[dataset.class_labels],
        ),
        DataTuple(
            x=data[dataset.discrete_features + dataset.cont_features],
            s=data[dataset.sens_attrs],
            y=data[dataset.class_labels],
        ),
    )
