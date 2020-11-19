"""Functions for synthetic data."""
from typing import Tuple

import numpy as np
import pandas as pd
from ethicml import Dataset, DataTuple
from sklearn import preprocessing


def make_x_bar(num_features: int, n: int, random_state: np.random.Generator) -> np.ndarray:
    """Initial potential."""
    x_tilde = random_state.normal(0, 1, size=(n, 1))
    x_tilde = np.zeros_like(x_tilde)

    return np.stack([random_state.normal(l, 1, num_features) for l in x_tilde])


def make_s(alpha: float, n: int, random_state: np.random.Generator, binary_s: bool) -> np.ndarray:
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
    return np.around(
        np.add(
            dx[:, :n],
            (xi * ((s[:, np.newaxis] * 2) - 1)) + np.random.normal(0, 0.05, dx[:, :n].shape),
        ),
        2,
    )


def make_y(y_bar: np.ndarray, s: pd.DataFrame, beta: float) -> pd.DataFrame:
    """Go from y_bar to Y."""
    y_bar = y_bar + beta * (s.values * 2 - 1)

    threshold = y_bar.quantile([0.75]).values[0][0]
    print(threshold)
    return (y_bar > threshold).astype(int).rename(columns={"y_bar": "y"})


def third_way_data(
    *,
    seed: int,
    num_features: int,
    num_samples: int,
    alpha: float,
    gamma: float,
    random_shift: int,
    binary_s: int,
    beta: float = 0.2,
    xi: float = 0.1,
) -> Tuple[Dataset, DataTuple, DataTuple]:
    """Generate very simple X data."""
    num_gen = np.random.default_rng(seed)
    x_bar = make_x_bar(num_features=num_features * 3, n=num_samples, random_state=num_gen)

    s = make_s(alpha=alpha, n=num_samples, random_state=num_gen, binary_s=binary_s == 1)
    s_df = pd.DataFrame(s, columns=["sens"])

    counterfactual_s = np.ones_like(s) - s if binary_s == 1 else np.zeros_like(s) - s
    counterfactual_s_df = pd.DataFrame(counterfactual_s, columns=["sens"])

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
    dx_df = pd.DataFrame(dx, columns=[f"dx_{i}" for i in range(dx.shape[1])])
    counterfactual_dx_df = pd.DataFrame(
        counterfactual_dx, columns=[f"dx_{i}" for i in range(dx.shape[1])]
    )

    x = make_x(dx, s, xi=xi, n=num_features)
    counterfactual_x = make_x(counterfactual_dx, counterfactual_s, xi=xi, n=num_features)
    x_df = pd.DataFrame(x, columns=[f"x_{i}" for i in range(x.shape[1])])
    counterfactual_x_df = pd.DataFrame(
        counterfactual_x, columns=[f"x_{i}" for i in range(x.shape[1])]
    )

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
    y_true_threshold = y_true_bar.quantile([0.75]).values[0][0]

    (y_true_bar > y_true_threshold).astype(int)
    (counterfactual_y_true_bar > y_true_threshold).astype(int)

    y_bar = pd.DataFrame(x_df.mean(axis=1), columns=["y_bar"])
    counterfactual_y_bar = pd.DataFrame(counterfactual_x_df.mean(axis=1), columns=["y_bar"])

    y = make_y(y_bar, s_df, beta=beta)
    counterfactual_y = make_y(counterfactual_y_bar, counterfactual_s_df, beta=beta)

    y_df = pd.DataFrame(y, columns=["outcome"])
    cf_y_df = pd.DataFrame(counterfactual_y, columns=["outcome"])

    data = pd.concat([x_df, s_df, y_df], axis=1).sort_index()
    counterfactual_data = pd.concat(
        [counterfactual_x_df, counterfactual_s_df, cf_y_df],
        axis=1,
    ).sort_index()

    idx = num_gen.permutation(data.index)

    data = data.reindex(idx).reset_index(drop=True)
    counterfactual_data = counterfactual_data.reindex(idx).reset_index(drop=True)

    dataset = Dataset(
        name=f"SimpleX",
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
    )
