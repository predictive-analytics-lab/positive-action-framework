"""Base Data Module."""
from typing import List

from ethicml import Dataset, DataTuple
from pytorch_lightning import LightningDataModule

from src.data_modules.dataset_utils import grouped_features_indexes


class BaseDataModule(LightningDataModule):
    """Simple 1d, configurable, data."""

    def __init__(self) -> None:
        super().__init__()
        self._columns = None
        self._cf_available = None
        self._x_dim = None
        self._num_s = None
        self._s_dim = None
        self.feature_groups = None

    @property
    def column_names(self) -> List[str]:
        assert self._columns is not None
        return self._columns

    @column_names.setter
    def column_names(self, col_names) -> None:
        self._columns = [f"{col}" for col in col_names]

    @property
    def cf_available(self) -> bool:
        assert self._cf_available is not None
        return self._cf_available

    @cf_available.setter
    def cf_available(self, true_cf_available: bool):
        self._cf_available = true_cf_available

    @property
    def data_dim(self) -> int:
        assert self._x_dim is not None
        return self._x_dim

    @data_dim.setter
    def data_dim(self, dim: int) -> None:
        self._x_dim = dim

    @property
    def num_s(self) -> int:
        assert self._num_s is not None
        return self._num_s

    @num_s.setter
    def num_s(self, dim: int) -> None:
        self._num_s = dim

    @property
    def s_dim(self) -> int:
        assert self._s_dim is not None
        return self._s_dim

    @s_dim.setter
    def s_dim(self, dim: int) -> None:
        self._s_dim = dim

    def make_feature_groups(self, dataset: Dataset, data: DataTuple) -> None:
        """Make feature groups for reconstruction."""
        disc_features = [feat for feat in dataset.discrete_features if feat in data.x.columns]
        self.disc_features = disc_features

        cont_features = [feat for feat in dataset.cont_features if feat in data.x.columns]
        self.cont_features = cont_features
        self.feature_groups = dict(discrete=grouped_features_indexes(self.disc_features))
