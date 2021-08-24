"""Base Data Module."""
from __future__ import annotations
from abc import abstractmethod
import warnings
from typing import NamedTuple

from ethicml import Dataset, DataTuple
from kit import implements
import numpy as np
import pandas as pd
from pytorch_lightning import LightningDataModule
from pytorch_lightning.loggers import WandbLogger
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
import numpy.typing as npt
from paf.base_templates.dataset_utils import grouped_features_indexes
from paf.plotting import label_plot

warnings.simplefilter(action='ignore', category=RuntimeWarning)


class BaseDataModule(LightningDataModule):
    """Simple 1d, configurable, data."""

    def __init__(self) -> None:
        super().__init__()
        self._cf_available: bool | None = None
        self._columns: list[str] | None = None
        self._card_s: int | None = None
        self._out_cols: list[str] | None = None
        self._s_dim: tuple[int, ...] | None = None
        self._y_dim: tuple[int, ...] | None = None
        self._test_tuple: DataTuple | None = None
        self._train_tuple: DataTuple | None = None
        self._x_dim: tuple[int, ...] | None = None
        self._feature_groups: dict[str, list[slice]] | None = None
        self._dataset: Dataset | None = None

    @property
    def outcome_columns(self) -> list[str]:
        assert self._out_cols is not None
        return self._out_cols

    @outcome_columns.setter
    def outcome_columns(self, out_cols: pd.Index) -> None:
        self._out_cols = [f"{col}" for col in out_cols]

    @property
    def feature_groups(self) -> dict[str, list[slice]]:
        assert self._feature_groups is not None
        return self._feature_groups

    @feature_groups.setter
    def feature_groups(self, feat_groups: dict[str, list[slice]]) -> None:
        self._feature_groups = feat_groups

    @property
    def column_names(self) -> list[str]:
        assert self._columns is not None
        return self._columns

    @column_names.setter
    def column_names(self, col_names: pd.Index) -> None:
        self._columns = [f"{col}" for col in col_names]

    @property
    def cf_available(self) -> bool:
        assert self._cf_available is not None
        return self._cf_available

    @cf_available.setter
    def cf_available(self, true_cf_available: bool) -> None:
        self._cf_available = true_cf_available

    @property
    def dataset(self) -> Dataset:
        assert self._dataset is not None
        return self._dataset

    @dataset.setter
    def dataset(self, dataset: Dataset) -> None:
        self._dataset = dataset

    @property
    def data_dim(self) -> tuple[int, ...]:
        assert self._x_dim is not None
        return self._x_dim

    @data_dim.setter
    def data_dim(self, dim: tuple[int, ...]) -> None:
        self._x_dim = dim

    @property
    def card_s(self) -> int:
        assert self._card_s is not None
        return self._card_s

    @card_s.setter
    def card_s(self, dim: int) -> None:
        self._card_s = dim

    @property
    def dim_s(self) -> tuple[int, ...]:
        assert self._s_dim is not None
        return self._s_dim

    @dim_s.setter
    def dim_s(self, dim: tuple[int, ...]) -> None:
        self._s_dim = dim

    @property
    def dim_y(self) -> tuple[int, ...]:
        assert self._y_dim is not None
        return self._y_dim

    @dim_y.setter
    def dim_y(self, dim: tuple[int, ...]) -> None:
        self._y_dim = dim

    def make_feature_groups(self, dataset: Dataset, data: DataTuple) -> None:
        """Make feature groups for reconstruction."""
        disc_features = [feat for feat in dataset.discrete_features if feat in data.x.columns]
        self.disc_features = disc_features

        cont_features = [feat for feat in dataset.continuous_features if feat in data.x.columns]
        self.cont_features = cont_features
        self.feature_groups = dict(discrete=grouped_features_indexes(self.disc_features))

    @property
    def train_datatuple(self) -> DataTuple:
        assert self._train_tuple is not None
        return self._train_tuple

    @train_datatuple.setter
    def train_datatuple(self, datatuple: DataTuple) -> None:
        self._train_tuple = datatuple

    @property
    def true_train_datatuple(self) -> DataTuple:
        assert self._true_train_tuple is not None
        assert self.cf_available
        return self._true_train_tuple

    @true_train_datatuple.setter
    def true_train_datatuple(self, datatuple: DataTuple) -> None:
        self._true_train_tuple = datatuple

    @property
    def test_datatuple(self) -> DataTuple:
        assert self._test_tuple is not None
        return self._test_tuple

    @test_datatuple.setter
    def test_datatuple(self, datatuple: DataTuple) -> None:
        self._test_tuple = datatuple

    @property
    def true_test_data(self) -> DataTuple:
        assert self._true_test_tuple is not None
        assert self.cf_available
        return self._true_test_tuple

    @true_test_data.setter
    def true_test_data(self, datatuple: DataTuple) -> None:
        self._true_test_tuple = datatuple

    @abstractmethod
    def _train_dataloader(self, shuffle: bool = True, drop_last: bool = True) -> DataLoader:
        ...

    @abstractmethod
    def _val_dataloader(self, shuffle: bool = True, drop_last: bool = True) -> DataLoader:
        ...

    @abstractmethod
    def _test_dataloader(self, shuffle: bool = True, drop_last: bool = True) -> DataLoader:
        ...

    @implements(LightningDataModule)
    def train_dataloader(self, shuffle: bool = True, drop_last: bool = True) -> DataLoader:
        return self._train_dataloader(shuffle, drop_last)

    @implements(LightningDataModule)
    def val_dataloader(self, shuffle: bool = False, drop_last: bool = False) -> DataLoader:
        return self._val_dataloader(shuffle, drop_last)

    @implements(LightningDataModule)
    def test_dataloader(self, shuffle: bool = False, drop_last: bool = False) -> DataLoader:
        return self._test_dataloader(shuffle, drop_last)

    def scale_and_split(
        self,
        datatuple: DataTuple,
        dataset: Dataset,
        train_indices: npt.NDArray[np.int_],
        val_indices: npt.NDArray[np.int_],
        test_indices: npt.NDArray[np.int_],
    ) -> ScaleSplitOut:
        """Scale a datatuple and split to train/test."""
        train = DataTuple(
            x=datatuple.x.iloc[train_indices].reset_index(drop=True),  # type: ignore[call-overload]
            s=datatuple.s.iloc[train_indices].reset_index(drop=True),  # type: ignore[call-overload]
            y=datatuple.y.iloc[train_indices].reset_index(drop=True),  # type: ignore[call-overload]
        )
        val = DataTuple(
            x=datatuple.x.iloc[val_indices].reset_index(drop=True),  # type: ignore[call-overload]
            s=datatuple.s.iloc[val_indices].reset_index(drop=True),  # type: ignore[call-overload]
            y=datatuple.y.iloc[val_indices].reset_index(drop=True),  # type: ignore[call-overload]
        )
        test = DataTuple(
            x=datatuple.x.iloc[test_indices].reset_index(drop=True),  # type: ignore[call-overload]
            s=datatuple.s.iloc[test_indices].reset_index(drop=True),  # type: ignore[call-overload]
            y=datatuple.y.iloc[test_indices].reset_index(drop=True),  # type: ignore[call-overload]
        )

        scaler = MinMaxScaler()
        scaler = scaler.fit(train.x[dataset.continuous_features])
        train.x[dataset.continuous_features] = scaler.transform(
            train.x[dataset.continuous_features]
        )
        val.x[dataset.continuous_features] = (
            scaler.transform(val.x[dataset.continuous_features])
            if val.x.shape[0] > 0
            else val.x[dataset.continuous_features]
        )
        test.x[dataset.continuous_features] = scaler.transform(test.x[dataset.continuous_features])
        return ScaleSplitOut(train=train, val=val, test=test)

    def make_data_plots(self, cf_available: bool, logger: WandbLogger | None) -> None:
        """Make plots of the data."""
        try:
            label_plot(self.train_datatuple, logger, "train")
        except (IndexError, KeyError):
            pass
        try:
            label_plot(self.test_datatuple, logger, "test")
        except (IndexError, KeyError):
            pass
        if cf_available and self.best_guess is not None:
            try:
                label_plot(
                    self.factual_data.replace(y=self.best_guess.hard.to_frame()),
                    logger,
                    "best_guess",
                )
                label_plot(self.cf_train, logger, "cf_train")
                label_plot(self.cf_test, logger, "cf_test")
                label_plot(self.s0_s0, logger, "s0_s0")
                label_plot(self.s0_s1, logger, "s0_s1")
                label_plot(self.s1_s0, logger, "s1_s0")
                label_plot(self.s1_s1, logger, "s1_s1")
            except (IndexError, KeyError):
                pass


class ScaleSplitOut(NamedTuple):
    train: DataTuple
    val: DataTuple
    test: DataTuple
