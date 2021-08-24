"""Functions for synthetic data."""
from __future__ import annotations
import logging

from ethicml import Dataset, DataTuple
import numpy as np
import pandas as pd
import scipy
from sklearn import preprocessing

log = logging.getLogger(__name__)

import numpy.typing as npt


def make_x_bar(
    num_features: int, n: int, random_state: np.random.Generator
) -> npt.NDArray[np.float_]:
    """Initial potential."""
    x_tilde = random_state.uniform(0, 1, size=(n, num_features))

    return np.stack([scipy.stats.norm.ppf(l, loc=0, scale=1) for l in x_tilde])


def make_s(
    alpha: float, n: int, random_state: np.random.Generator, binary_s: bool
) -> npt.NDArray[np.float_]:
    """Set S."""
    if binary_s:
        return random_state.binomial(1, alpha, n)
    else:
        return random_state.uniform(-1, 1, n)


def make_dx(x_bar: np.ndarray, s: np.ndarray, gamma: float, binary_s: bool) -> np.ndarray:
    """Skew the data replicating life experience."""
    if binary_s:
        return np.add(x_bar[:, : x_bar.shape[1]], (gamma * ((s[:, np.newaxis] * 2) - 1)))
    else:
        return np.add(x_bar[:, : x_bar.shape[1]], (gamma * (s[:, np.newaxis])))


def make_x(dx: np.ndarray, s: np.ndarray, xi: float, n: int) -> np.ndarray:
    """Make observations of the data."""
    rng = np.random.RandomState(0)
    return np.around(
        np.add(
            dx[:, :n],
            (xi * ((s[:, np.newaxis] * 2) - 1)) + rng.normal(0, 0.05, dx[:, :n].shape),
        ),
        2,
    )


def make_y(
    y_bar: pd.DataFrame,
    s: pd.DataFrame,
    beta: float,
    acceptance_rate: float,
    threshold: float | None = None,
) -> tuple[pd.DataFrame, float]:
    """Go from y_bar to Y."""
    y_bar = y_bar + beta * (s.values * 2 - 1)

    if threshold is None:
        threshold = y_bar.quantile([1 - acceptance_rate]).values[0][0]
        print(threshold)
    return (y_bar > threshold).astype(int).rename(columns={"y_bar": "y"}), threshold


def third_way_data(
    *,
    seed: int,
    num_hidden_features: int,
    num_features: int,
    num_samples: int,
    acceptance_rate: float,
    alpha: float,
    gamma: float,
    random_shift: int,
    binary_s: int,
    beta: float,
    xi: float,
) -> tuple[Dataset, DataTuple, DataTuple, DataTuple, DataTuple, DataTuple, DataTuple, DataTuple]:
    """Generate very simple X data."""
    num_gen = np.random.default_rng(seed)
    x_bar = make_x_bar(num_features=num_hidden_features, n=num_samples, random_state=num_gen)

    s = make_s(alpha=alpha, n=num_samples, random_state=num_gen, binary_s=binary_s == 1)
    s_df = pd.DataFrame(s, columns=["sens"])
    counterfactual_s: np.ndarray = np.ones_like(s) - s if binary_s == 1 else np.zeros_like(s) - s
    cf_s_df = pd.DataFrame(counterfactual_s, columns=["sens"])
    s0 = np.zeros_like(s)
    s0_df = pd.DataFrame(s0, columns=["sens"])
    s1 = np.ones_like(s)
    s1_df = pd.DataFrame(s1, columns=["sens"])

    if random_shift == 0:
        temp_s = s
        tmp_cf_s = counterfactual_s
    else:
        temp_s = make_s(alpha, num_samples, np.random.default_rng(seed + 9), binary_s=binary_s == 1)
        tmp_cf_s = (
            np.ones_like(temp_s) - temp_s if binary_s == 1 else np.zeros_like(temp_s) - temp_s
        )

    dx = make_dx(x_bar=x_bar, s=temp_s, gamma=gamma, binary_s=binary_s == 1)
    counterfactual_dx = make_dx(x_bar, tmp_cf_s, gamma=gamma, binary_s=binary_s == 1)
    s0_dx = make_dx(x_bar, s0, gamma=gamma, binary_s=binary_s == 1)
    s1_dx = make_dx(x_bar, s1, gamma=gamma, binary_s=binary_s == 1)

    x = make_x(dx, s, xi=xi, n=num_features)
    x_df = pd.DataFrame(x, columns=[f"x_{i}" for i in range(x.shape[1])])
    counterfactual_x = make_x(counterfactual_dx, counterfactual_s, xi=xi, n=num_features)
    cf_x_df = pd.DataFrame(counterfactual_x, columns=[f"x_{i}" for i in range(x.shape[1])])
    s0_x = make_x(s0_dx, s0, xi=xi, n=num_features)
    s0_x_df = pd.DataFrame(s0_x, columns=[f"x_{i}" for i in range(x.shape[1])])
    s1_x = make_x(s1_dx, s1, xi=xi, n=num_features)
    s1_x_df = pd.DataFrame(s1_x, columns=[f"x_{i}" for i in range(x.shape[1])])

    w = np.ones(dx.shape[1]) / dx.shape[1]  # np.random.normal(0, 1, dx.shape[1])

    min_max_scaler = preprocessing.StandardScaler()

    y_true_bar = pd.DataFrame(
        [np.dot(w, _x) for _x in min_max_scaler.fit_transform(dx)],
        columns=["y_true"],
    )
    counterfactual_y_true_bar = pd.DataFrame(
        [np.dot(w, _x) for _x in min_max_scaler.fit_transform(counterfactual_dx)],
        columns=["y_true"],
    )
    s0_y_true_bar = pd.DataFrame(
        [np.dot(w, _x) for _x in min_max_scaler.fit_transform(s0_dx)],
        columns=["y_true"],
    )
    s1_y_true_bar = pd.DataFrame(
        [np.dot(w, _x) for _x in min_max_scaler.fit_transform(s1_dx)],
        columns=["y_true"],
    )

    y_true_threshold = y_true_bar.quantile([0.75]).values[0][0]

    y_true = (y_true_bar > y_true_threshold).astype(int)
    (counterfactual_y_true_bar > y_true_threshold).astype(int)
    (s0_y_true_bar > y_true_threshold).astype(int)
    (s1_y_true_bar > y_true_threshold).astype(int)

    y_bar = pd.DataFrame(x_df.mean(axis=1), columns=["y_bar"])
    cf_y_bar = pd.DataFrame(cf_x_df.mean(axis=1), columns=["y_bar"])
    s0_y_bar = pd.DataFrame(s0_x_df.mean(axis=1), columns=["y_bar"])
    s1_y_bar = pd.DataFrame(s1_x_df.mean(axis=1), columns=["y_bar"])

    y_df, threshold = make_y(y_bar, s_df, beta=beta, acceptance_rate=acceptance_rate)
    y_df = y_df.rename(columns={"y": "outcome"})
    cf_y_df, _ = make_y(
        cf_y_bar, cf_s_df, beta=beta, acceptance_rate=acceptance_rate, threshold=threshold
    )
    cf_y_df = cf_y_df.rename(columns={"y": "outcome"})
    s1_0_s2_0_y_df, _ = make_y(
        s0_y_bar, s0_df, beta=beta, acceptance_rate=acceptance_rate, threshold=threshold
    )
    s1_0_s2_0_y_df = s1_0_s2_0_y_df.rename(columns={"y": "outcome"})
    s1_0_s2_1_y_df, _ = make_y(
        s0_y_bar, s1_df, beta=beta, acceptance_rate=acceptance_rate, threshold=threshold
    )
    s1_0_s2_1_y_df = s1_0_s2_1_y_df.rename(columns={"y": "outcome"})
    s1_1_s2_0_y_df, _ = make_y(
        s1_y_bar, s0_df, beta=beta, acceptance_rate=acceptance_rate, threshold=threshold
    )
    s1_1_s2_0_y_df = s1_1_s2_0_y_df.rename(columns={"y": "outcome"})
    s1_1_s2_1_y_df, _ = make_y(
        s1_y_bar, s1_df, beta=beta, acceptance_rate=acceptance_rate, threshold=threshold
    )
    s1_1_s2_1_y_df = s1_1_s2_1_y_df.rename(columns={"y": "outcome"})

    data = pd.concat([x_df, s_df, y_df, y_true], axis=1).sort_index()
    counterfactual_data = pd.concat([cf_x_df, cf_s_df, cf_y_df], axis=1).sort_index()
    s1_0_s2_0_data = pd.concat([s0_x_df, s0_df, s1_0_s2_0_y_df], axis=1).sort_index()
    s1_0_s2_1_data = pd.concat([s0_x_df, s0_df, s1_0_s2_1_y_df], axis=1).sort_index()
    s1_1_s2_0_data = pd.concat([s1_x_df, s0_df, s1_1_s2_0_y_df], axis=1).sort_index()
    s1_1_s2_1_data = pd.concat([s1_x_df, s0_df, s1_1_s2_1_y_df], axis=1).sort_index()

    idx = num_gen.permutation(data.index)

    data = data.reindex(idx).reset_index(drop=True)  # type: ignore[call-arg]
    counterfactual_data = counterfactual_data.reindex(idx).reset_index(drop=True)  # type: ignore[call-arg]
    s1_0_s2_0_data = s1_0_s2_0_data.reindex(idx).reset_index(drop=True)  # type: ignore[call-arg]
    s1_0_s2_1_data = s1_0_s2_1_data.reindex(idx).reset_index(drop=True)  # type: ignore[call-arg]
    s1_1_s2_0_data = s1_1_s2_0_data.reindex(idx).reset_index(drop=True)  # type: ignore[call-arg]
    s1_1_s2_1_data = s1_1_s2_1_data.reindex(idx).reset_index(drop=True)  # type: ignore[call-arg]

    dataset = Dataset(
        name=f"ThirdWay",
        num_samples=num_samples,
        filename_or_path="none",
        features=[f"x_{i}" for i in range(num_features)] + ["sens"],
        cont_features=[f"x_{i}" for i in range(num_features)],
        sens_attr_spec="sens",
        s_prefix="sens",
        class_label_spec=f"outcome",
        class_label_prefix="outcome",
        discrete_only=False,
    )

    return (
        dataset,
        DataTuple(x=data[x_df.columns], s=data[s_df.columns], y=data[y_df.columns]),
        DataTuple(
            x=counterfactual_data[x_df.columns],
            s=counterfactual_data[s_df.columns],
            y=counterfactual_data[y_df.columns],
        ),
        DataTuple(x=data[x_df.columns], s=data[s_df.columns], y=data[y_true.columns]),
        DataTuple(
            x=s1_0_s2_0_data[x_df.columns],
            s=s1_0_s2_0_data[s_df.columns],
            y=s1_0_s2_0_data[y_df.columns],
        ),
        DataTuple(
            x=s1_0_s2_1_data[x_df.columns],
            s=s1_0_s2_1_data[s_df.columns],
            y=s1_0_s2_1_data[y_df.columns],
        ),
        DataTuple(
            x=s1_1_s2_0_data[x_df.columns],
            s=s1_1_s2_0_data[s_df.columns],
            y=s1_1_s2_0_data[y_df.columns],
        ),
        DataTuple(
            x=s1_1_s2_1_data[x_df.columns],
            s=s1_1_s2_1_data[s_df.columns],
            y=s1_1_s2_1_data[y_df.columns],
        ),
    )
