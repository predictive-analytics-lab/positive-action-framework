"""Base Data Module."""
from typing import Dict, List, Optional

import pandas as pd
from ethicml import Dataset, DataTuple
from pytorch_lightning import LightningDataModule

from src.data_modules.dataset_utils import grouped_features_indexes


class BaseDataModule(LightningDataModule):
    """Simple 1d, configurable, data."""

    def __init__(self) -> None:
        super().__init__()
        self._cf_available: Optional[bool] = None
        self._columns: Optional[List[str]] = None
        self._num_s: Optional[int] = None
        self._out_cols: Optional[List[str]] = None
        self._s_dim: Optional[int] = None
        self._y_dim: Optional[int] = None
        self._test_tuple: Optional[DataTuple] = None
        self._train_tuple: Optional[DataTuple] = None
        self._x_dim: Optional[int] = None
        self._feature_groups: Optional[Dict[str, List[slice]]] = None

    @property
    def outcome_columns(self) -> List[str]:
        assert self._out_cols is not None
        return self._out_cols

    @outcome_columns.setter
    def outcome_columns(self, out_cols: pd.Index) -> None:
        self._out_cols = [f"{col}" for col in out_cols]

    @property
    def feature_groups(self) -> Dict[str, List[slice]]:
        assert self._feature_groups is not None
        return self._feature_groups

    @feature_groups.setter
    def feature_groups(self, feat_groups: Dict[str, List[slice]]) -> None:
        self._feature_groups = feat_groups

    @property
    def column_names(self) -> List[str]:
        assert self._columns is not None
        return self._columns  # type: ignore[unreachable]

    @column_names.setter
    def column_names(self, col_names: pd.Index) -> None:
        self._columns = [f"{col}" for col in col_names]

    @property
    def cf_available(self) -> bool:
        assert self._cf_available is not None
        return self._cf_available  # type: ignore[unreachable]

    @cf_available.setter
    def cf_available(self, true_cf_available: bool) -> None:
        self._cf_available = true_cf_available

    @property
    def data_dim(self) -> int:
        assert self._x_dim is not None
        return self._x_dim  # type: ignore[unreachable]

    @data_dim.setter
    def data_dim(self, dim: int) -> None:
        self._x_dim = dim

    @property
    def num_s(self) -> int:
        assert self._num_s is not None
        return self._num_s  # type: ignore[unreachable]

    @num_s.setter
    def num_s(self, dim: int) -> None:
        self._num_s = dim

    @property
    def s_dim(self) -> int:
        assert self._s_dim is not None
        return self._s_dim  # type: ignore[unreachable]

    @s_dim.setter
    def s_dim(self, dim: int) -> None:
        self._s_dim = dim

    @property
    def y_dim(self) -> int:
        assert self._y_dim is not None
        return self._y_dim  # type: ignore[unreachable]

    @y_dim.setter
    def y_dim(self, dim: int) -> None:
        self._y_dim = dim

    def make_feature_groups(self, dataset: Dataset, data: DataTuple) -> None:
        """Make feature groups for reconstruction."""
        disc_features = [feat for feat in dataset.discrete_features if feat in data.x.columns]
        self.disc_features = disc_features

        cont_features = [feat for feat in dataset.continuous_features if feat in data.x.columns]
        self.cont_features = cont_features
        self.feature_groups = dict(discrete=grouped_features_indexes(self.disc_features))

    @property
    def train_data(self) -> DataTuple:
        assert self._train_tuple is not None
        return self._train_tuple  # type: ignore[unreachable]

    @train_data.setter
    def train_data(self, datatuple: DataTuple) -> None:
        self._train_tuple = datatuple

    @property
    def true_train_data(self) -> DataTuple:
        assert self._true_train_tuple is not None
        assert self.cf_available
        return self._true_train_tuple  # type: ignore[unreachable]

    @true_train_data.setter
    def true_train_data(self, datatuple: DataTuple) -> None:
        self._true_train_tuple = datatuple

    @property
    def test_data(self) -> DataTuple:
        assert self._test_tuple is not None
        return self._test_tuple  # type: ignore[unreachable]

    @test_data.setter
    def test_data(self, datatuple: DataTuple) -> None:
        self._test_tuple = datatuple

    @property
    def true_test_data(self) -> DataTuple:
        assert self._true_test_tuple is not None
        assert self.cf_available
        return self._true_test_tuple  # type: ignore[unreachable]

    @true_test_data.setter
    def true_test_data(self, datatuple: DataTuple) -> None:
        self._true_test_tuple = datatuple
