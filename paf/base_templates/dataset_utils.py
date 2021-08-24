"""Torch Dataset wrapper for EthicML."""
from __future__ import annotations
from itertools import groupby
from typing import Iterator, NamedTuple

from ethicml import DataTuple, compute_instance_weights
from ethicml.implementations.pytorch_common import _get_info
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


class Batch(NamedTuple):
    x: Tensor
    s: Tensor
    y: Tensor
    iw: Tensor


class CfBatch(NamedTuple):
    x: Tensor
    s: Tensor
    y: Tensor
    cfx: Tensor
    cfs: Tensor
    cfy: Tensor
    iw: Tensor


def group_features(disc_feats: list[str]) -> Iterator[tuple[str, Iterator[str]]]:
    """Group discrete features names according to the first segment of their name."""

    def _first_segment(feature_name: str) -> str:
        return feature_name.split("_")[0]

    return groupby(disc_feats, _first_segment)


def grouped_features_indexes(disc_feats: list[str]) -> list[slice]:
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


class DataTupleDatasetBase(Dataset):
    """Wrapper for EthicML datasets."""

    def __init__(self, dataset: DataTuple, disc_features: list[str], cont_features: list[str]):
        """Create DataTupleDataset."""
        disc_features = [feat for feat in disc_features if feat in dataset.x.columns]
        self.disc_features = disc_features

        cont_features = [feat for feat in cont_features if feat in dataset.x.columns]
        self.cont_features = cont_features
        self.feature_groups = dict(discrete=grouped_features_indexes(self.disc_features))

        self.x_disc = dataset.x[self.disc_features].astype("float32").to_numpy()
        self.x_cont = dataset.x[self.cont_features].astype("float32").to_numpy()

        _, self.s, self.num, self.xdim, self.sdim, self.x_names, self.s_names = _get_info(dataset)

        self.y = dataset.y.astype("float32").to_numpy()

        self.ydim = dataset.y.shape[1]
        self.y_names = dataset.y.columns

    def __len__(self) -> int:
        return self.s.shape[0]

    def _x(self, index: int) -> Tensor:
        x_disc = self.x_disc[index]
        x_cont = self.x_cont[index]
        x = np.concatenate([x_disc, x_cont], axis=0, dtype=np.float32)
        x = torch.from_numpy(x)
        return x

    def _s(self, index: int) -> Tensor:
        s = self.s[index]
        return torch.from_numpy(s).squeeze()

    def _y(self, index: int) -> Tensor:
        y = self.y[index]
        return torch.from_numpy(y).squeeze()


class DataTupleDataset(DataTupleDatasetBase):
    """Wrapper for EthicML datasets."""

    def __init__(self, dataset: DataTuple, disc_features: list[str], cont_features: list[str]):
        super().__init__(dataset, disc_features, cont_features)
        self.instance_weight = torch.tensor(
            compute_instance_weights(dataset)["instance weights"].values
        )

    def __getitem__(self, index: int) -> Batch:
        return Batch(
            x=self._x(index), s=self._s(index), y=self._y(index), iw=self.instance_weight[index]
        )


class CFDataTupleDataset(DataTupleDatasetBase):
    """Wrapper for EthicML datasets."""

    def __init__(
        self,
        dataset: DataTuple,
        cf_dataset: DataTuple,
        disc_features: list[str],
        cont_features: list[str],
    ):
        """Create DataTupleDataset."""
        super().__init__(dataset, disc_features, cont_features)
        self.cf_x_disc, self.cf_x_cont, self.cf_s, self.cf_y = self.split_tuple(cf_dataset)
        self.instance_weight = torch.tensor(
            compute_instance_weights(dataset)["instance weights"].values
        )

    def split_tuple(
        self, datatuple: DataTuple
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split a datatuple to components."""
        x_disc = datatuple.x[self.disc_features].astype("float32").to_numpy()
        x_cont = datatuple.x[self.cont_features].astype("float32").to_numpy()
        s = datatuple.s.astype("float32").to_numpy()
        y = datatuple.y.astype("float32").to_numpy()
        return x_disc, x_cont, s, y

    def _make_from_arr(self, np_array: np.ndarray, index: int) -> Tensor:
        np_a = np_array[index]
        return torch.from_numpy(np_a).squeeze()

    def _make_x(self, disc: np.ndarray, cont: np.ndarray, index: int) -> Tensor:
        x_disc = disc[index]
        x_cont = cont[index]
        x = np.concatenate([x_disc, x_cont], axis=0, dtype=np.float32)
        x = torch.from_numpy(x)
        return x

    def _cf_x(self, index: int) -> Tensor:
        return self._make_x(self.cf_x_disc, self.cf_x_cont, index)

    def _cf_s(self, index: int) -> Tensor:
        return self._make_from_arr(self.cf_s, index)

    def _cf_y(self, index: int) -> Tensor:
        return self._make_from_arr(self.cf_y, index)

    def __getitem__(self, index: int) -> CfBatch:
        return CfBatch(
            x=super()._x(index),
            s=super()._s(index),
            y=super()._y(index),
            cfx=self._cf_x(index),
            cfs=self._cf_s(index),
            cfy=self._cf_y(index),
            iw=self.instance_weight[index],
        )
