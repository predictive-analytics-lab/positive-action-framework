"""Functions for synthetic data."""

from __future__ import annotations

from ethicml import Dataset, DataTuple
import numpy as np
import pandas as pd


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


def make_x(x_bar: np.ndarray, s: np.ndarray, gamma: float, binary_s: bool) -> np.ndarray:
    """Skew the data replicating life experience."""
    if binary_s:
        return np.add(x_bar[:, : x_bar.shape[1]], (gamma * ((s[:, np.newaxis] * 2) - 1)))
    else:
        return np.add(x_bar[:, : x_bar.shape[1]], (gamma * (s[:, np.newaxis])))


def simple_x_data(
    *, seed: int, num_samples: int, alpha: float, gamma: float, random_shift: int, binary_s: int
) -> tuple[Dataset, DataTuple, DataTuple, DataTuple]:
    """Generate very simple X data."""
    num_gen = np.random.default_rng(seed)
    x_bar = make_x_bar(num_features=1, n=num_samples, random_state=num_gen)

    s = make_s(alpha=alpha, n=num_samples, random_state=num_gen, binary_s=binary_s == 1)
    s_df = pd.DataFrame(s, columns=["sens"])

    counterfactual_s: np.ndarray = np.ones_like(s) - s if binary_s == 1 else np.zeros_like(s) - s
    counterfactual_s_df = pd.DataFrame(counterfactual_s, columns=["sens"])

    outcome_placeholder = pd.DataFrame(num_gen.binomial(1, 0.5, len(s)), columns=["outcome"])

    if random_shift == 0:
        temp_s = s
        tmp_cf_s = counterfactual_s
    else:
        temp_s = make_s(alpha, num_samples, np.random.default_rng(seed + 9), binary_s=binary_s == 1)
        tmp_cf_s = (
            np.ones_like(temp_s) - temp_s if binary_s == 1 else np.zeros_like(temp_s) - temp_s
        )

    x = make_x(x_bar=x_bar, s=temp_s, gamma=gamma, binary_s=binary_s == 1)
    counterfactual_x = make_x(x_bar, tmp_cf_s, gamma=gamma, binary_s=binary_s == 1)
    x_df = pd.DataFrame(x, columns=[f"x_{i}" for i in range(x.shape[1])])
    counterfactual_x_df = pd.DataFrame(
        counterfactual_x, columns=[f"x_{i}" for i in range(x.shape[1])]
    )

    data = pd.concat([x_df, s_df, outcome_placeholder], axis=1).sort_index()
    counterfactual_data = pd.concat(
        [counterfactual_x_df, counterfactual_s_df, 1 - outcome_placeholder],
        axis=1,
    ).sort_index()

    idx = num_gen.permutation(data.index)  # type: ignore[arg-type]

    data = data.reindex(idx).reset_index(drop=True)  # type: ignore[call-arg]
    counterfactual_data = counterfactual_data.reindex(idx).reset_index(drop=True)  # type: ignore[call-arg]

    dataset = Dataset(
        name=f"SimpleX",
        num_samples=num_samples,
        filename_or_path="none",
        features=[f"x_{i}" for i in range(1)] + ["sens"],
        cont_features=[f"x_{i}" for i in range(1)],
        sens_attr_spec="sens",
        s_prefix="sens",
        class_label_spec=f"outcome",
        class_label_prefix="outcome",
        discrete_only=False,
    )

    return (
        dataset,
        DataTuple(x=data[x_df.columns], s=data[s_df.columns], y=data[outcome_placeholder.columns]),
        DataTuple(
            x=counterfactual_data[x_df.columns],
            s=counterfactual_data[s_df.columns],
            y=counterfactual_data[outcome_placeholder.columns],
        ),
        DataTuple(
            x=counterfactual_data[x_df.columns],
            s=counterfactual_data[s_df.columns],
            y=counterfactual_data[outcome_placeholder.columns],
        ),
    )
