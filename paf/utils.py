"""Utility functions."""
from __future__ import annotations
import collections
from typing import Any, MutableMapping
import warnings

import pandas as pd
import torch
from torch import Tensor
import numpy as np

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)

__all__ = [
    "flatten",
    "facct_mapper",
    "facct_mapper_2",
    "facct_mapper_outcomes",
    "HistoryPool",
    "Stratifier",
]


def flatten(
    dict_: MutableMapping[Any, Any], parent_key: str = "", sep: str = "."
) -> dict[Any, Any]:
    """Flatten a nested dictionary by separating the keys with `sep`."""
    items: list[Any] = []
    for key, value in dict_.items():
        new_key = parent_key + sep + key if parent_key else key
        if isinstance(value, collections.MutableMapping):
            items.extend(flatten(value, new_key, sep=sep).items())
        else:
            items.append((new_key, value))
    return dict(items)


def facct_mapper(facct_out: pd.Series) -> pd.Series:
    """Map from groups to outcomes."""
    lookup = {
        0: 5,
        1: 6,
        2: 3,
        3: 4,
        4: 7,
        5: 7,
        6: 3,
        7: 4,
        8: 7,
        9: 7,
        10: 1,
        11: 4,
        12: 7,
        13: 7,
        14: 1,
        15: 2,
        16: 8,
        17: 7,
        18: 8,
        19: 8,
        20: 8,
        21: 7,
        22: 8,
        23: 8,
        24: 8,
        25: 7,
        26: 8,
        27: 8,
        28: 8,
        29: 7,
        30: 1,
        31: 2,
    }

    return pd.Series({i: lookup[d] for i, d in enumerate(facct_out)})


def facct_mapper_2(facct_out: pd.Series) -> pd.Series:
    """Map from groups to outcomes."""
    lookup = {-1: 0, 1: 1, 2: 1, 3: 2, 4: 1, 5: 0, 6: 0, 7: 0, 8: 1}

    return pd.Series({i: lookup[d] for i, d in enumerate(facct_out)})


def facct_mapper_outcomes(mapped: pd.Series, fair: bool) -> pd.Series:
    """Make the final outcome."""
    lookup = {0: 0, 1: 1, 2: 1 if fair else 0}

    return pd.Series({i: lookup[d] for i, d in enumerate(mapped)})


class HistoryPool:
    def __init__(self, pool_size: int = 50):
        self.nb_samples = 0
        self.history_pool: list[Tensor] = []
        self.pool_sz = pool_size

    def push_and_pop(self, samples: Tensor) -> Tensor:
        samples_to_return = []
        for sample in samples:
            sample = torch.unsqueeze(sample, 0)
            if self.nb_samples < self.pool_sz:
                self.history_pool.append(sample)
                samples_to_return.append(sample)
                self.nb_samples += 1
            elif np.random.uniform(0, 1) > 0.5:
                rand_int = np.random.randint(0, self.pool_sz)
                temp_img = self.history_pool[rand_int].clone()
                self.history_pool[rand_int] = sample
                samples_to_return.append(temp_img)
            else:
                samples_to_return.append(sample)
        return torch.cat(samples_to_return, 0)


class Stratifier:
    def __init__(self, pool_size: int = 50):
        self.nb_samples = 0
        self.history_pool: list[Tensor] = []
        self.pool_sz = pool_size

    def push_and_pop(self, samples: Tensor) -> Tensor:
        samples_to_return = []
        for sample in samples:
            if len(samples_to_return) >= self.pool_sz:
                break
            sample = torch.unsqueeze(sample, 0)
            if self.nb_samples < self.pool_sz:
                self.history_pool.append(sample)
                samples_to_return.append(sample)
                self.nb_samples += 1
            elif np.random.uniform(0, 1) > 0.5:
                rand_int = np.random.randint(0, self.pool_sz)
                temp_img = self.history_pool[rand_int].clone()
                self.history_pool[rand_int] = sample
                samples_to_return.append(temp_img)
            else:
                samples_to_return.append(sample)
        while len(samples_to_return) < self.pool_sz:
            rand_int = np.random.randint(0, self.nb_samples)
            temp_img = self.history_pool[rand_int].clone()
            samples_to_return.append(temp_img)
        return torch.cat(samples_to_return, 0)
