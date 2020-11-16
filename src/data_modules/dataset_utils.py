"""Torch Dataset wrapper for EthicML."""

from itertools import groupby
from typing import Iterator, List, Tuple

import numpy as np
import torch
from ethicml import DataTuple
from ethicml.implementations.pytorch_common import _get_info
from torch import Tensor


def group_features(disc_feats: List[str]) -> Iterator[Tuple[str, Iterator[str]]]:
    """Group discrete features names according to the first segment of their name."""

    def _first_segment(feature_name: str) -> str:
        return feature_name.split("_")[0]

    return groupby(disc_feats, _first_segment)


def grouped_features_indexes(disc_feats: List[str]) -> List[slice]:
    """Group discrete features names according to the first segment of their name.

    Then return a list of their corresponding slices (assumes order is maintained).
    """
    group_iter = group_features(disc_feats)

    feature_slices = []
    start_idx = 0
    for _, group in group_iter:
        len_group = len(list(group))
        indexes = slice(start_idx, start_idx + len_group)
        feature_slices.append(indexes)
        start_idx += len_group

    return feature_slices


class DataTupleDataset:
    """Wrapper for EthicML datasets."""

    def __init__(
        self,
        dataset: DataTuple,
        cf_dataset: DataTuple,
        disc_features: List[str],
        cont_features: List[str],
    ):
        """Create DataTupleDataset."""
        disc_features = [feat for feat in disc_features if feat in dataset.x.columns]
        self.disc_features = disc_features

        cont_features = [feat for feat in cont_features if feat in dataset.x.columns]
        self.cont_features = cont_features
        self.feature_groups = dict(discrete=grouped_features_indexes(self.disc_features))

        self.x_disc = dataset.x[self.disc_features].to_numpy(dtype=np.float32)
        self.x_cont = dataset.x[self.cont_features].to_numpy(dtype=np.float32)

        self.cf_x_disc = cf_dataset.x[self.disc_features].to_numpy(dtype=np.float32)
        self.cf_x_cont = cf_dataset.x[self.cont_features].to_numpy(dtype=np.float32)

        _, self.s, self.num, self.xdim, self.sdim, self.x_names, self.s_names = _get_info(dataset)
        _, self.cf_s, _, _, _, _, _ = _get_info(cf_dataset)

        self.y = dataset.y.to_numpy(dtype=np.float32)
        self.cf_y = cf_dataset.y.to_numpy(dtype=np.float32)

        self.ydim = dataset.y.shape[1]
        self.y_names = dataset.y.columns

    def __len__(self) -> int:
        return self.s.shape[0]

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:

        x_disc = self.x_disc[index]
        x_cont = self.x_cont[index]

        cf_x_disc = self.cf_x_disc[index]
        cf_x_cont = self.cf_x_cont[index]

        s = self.s[index]
        cf_s = self.cf_s[index]

        y = self.y[index]
        cf_y = self.cf_y[index]

        x = np.concatenate([x_disc, x_cont], axis=0)
        x = torch.from_numpy(x)
        cf_x = np.concatenate([cf_x_disc, cf_x_cont], axis=0)
        cf_x = torch.from_numpy(cf_x)

        if x.shape == 1:
            x = x.squeeze(0)
            cf_x = cf_x.squeeze(0)

        s = torch.from_numpy(s).squeeze()
        cf_s = torch.from_numpy(cf_s).squeeze()

        y = torch.from_numpy(y).squeeze()
        cf_y = torch.from_numpy(cf_y).squeeze()

        return x, s, y, cf_x, cf_s, cf_y
