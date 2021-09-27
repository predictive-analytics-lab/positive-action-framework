"""Base Data Module."""
from __future__ import annotations
from abc import abstractmethod
from typing import NamedTuple
import warnings

from ethicml import Dataset, DataTuple, Prediction
from kit import implements
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import pytorch_lightning.loggers as pll
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

from paf.base_templates.dataset_utils import grouped_features_indexes
from paf.plotting import label_plot

__all__ = ["BaseDataModule", "ScaleSplitOut"]

warnings.simplefilter(action='ignore', category=RuntimeWarning)


class BaseDataModule(pl.LightningDataModule):
    """Simple 1d, configurable, data."""

    columns: list[str]
    card_s: int
    out_cols: list[str]
    s_dim: tuple[int, ...]
    y_dim: tuple[int, ...]
    test_tuple: DataTuple
    train_tuple: DataTuple
    x_dim: tuple[int, ...]
    feature_groups: dict[str, list[slice]]
    dataset: Dataset
    disc_features: list[str]
    cont_features: list[str]
    best_guess: Prediction | None
    factual_data: DataTuple
    card_y: int
    dims: tuple[int, ...]
    dim_s: tuple[int, ...]
    column_names: list[str]
    outcome_columns: list[str]
    train_datatuple: DataTuple
    val_datatuple: DataTuple
    test_datatuple: DataTuple
    cf_train_datatuple: DataTuple | None
    cf_val_datatuple: DataTuple | None
    cf_test_datatuple: DataTuple | None
    true_train_datatuple: DataTuple | None
    true_val_datatuple: DataTuple | None
    true_test_datatuple: DataTuple | None
    scaler: MinMaxScaler | None
    s0_s0: DataTuple | None
    s0_s1: DataTuple | None
    s1_s0: DataTuple | None
    s1_s1: DataTuple | None

    def __init__(self, *, cf_available: bool, seed: int, scaler: MinMaxScaler | None) -> None:
        super().__init__()
        self.cf_available = cf_available
        self.seed = seed
        self.train_indices: pd.Index[int] | None = None
        self.val_indices: pd.Index[int] | None = None
        self.test_indices: pd.Index[int] | None = None
        self.scaler = scaler

    def set_data_values(
        self,
        *,
        dataset: Dataset,
        factual_data: DataTuple,
        dts: ScaleSplitOut,
        best_guess: Prediction | None = None,
        true_dts: ScaleSplitOut | None = None,
        cf_dts: ScaleSplitOut | None = None,
        s0_s0: DataTuple | None = None,
        s0_s1: DataTuple | None = None,
        s1_s0: DataTuple | None = None,
        s1_s1: DataTuple | None = None,
    ) -> None:
        self.dataset = dataset
        self.best_guess = best_guess
        self.factual_data = factual_data
        self.card_s = factual_data.s.nunique().values[0]
        self.card_y = factual_data.y.nunique().values[0]
        self.dims = factual_data.x.shape[1:]
        self.dim_s = (1,) if factual_data.s.ndim == 1 else factual_data.s.shape[1:]
        self.column_names = [str(col) for col in factual_data.x.columns]
        self.outcome_columns = [str(col) for col in factual_data.y.columns]
        self.train_datatuple = dts.train
        self.val_datatuple = dts.val
        self.test_datatuple = dts.test
        self.true_train_datatuple = true_dts.train if true_dts is not None else None
        self.true_val_datatuple = true_dts.val if true_dts is not None else None
        self.true_test_datatuple = true_dts.test if true_dts is not None else None
        self.cf_train_datatuple = cf_dts.train if cf_dts is not None else None
        self.cf_val_datatuple = cf_dts.val if cf_dts is not None else None
        self.cf_test_datatuple = cf_dts.test if cf_dts is not None else None
        self.s0_s0 = s0_s0
        self.s0_s1 = s0_s1
        self.s1_s0 = s1_s0
        self.s1_s1 = s1_s1
        self.make_feature_groups(dataset=dataset, data=factual_data)

    def make_feature_groups(self, *, dataset: Dataset, data: DataTuple) -> None:
        """Make feature groups for reconstruction."""
        disc_features = [feat for feat in dataset.discrete_features if feat in data.x.columns]
        self.disc_features = disc_features

        cont_features = [feat for feat in dataset.continuous_features if feat in data.x.columns]
        self.cont_features = cont_features
        self.feature_groups = dict(discrete=grouped_features_indexes(self.disc_features))

    @abstractmethod
    def _train_dataloader(self, *, shuffle: bool = True, drop_last: bool = True) -> DataLoader:
        ...

    @abstractmethod
    def _val_dataloader(self, *, shuffle: bool = True, drop_last: bool = True) -> DataLoader:
        ...

    @abstractmethod
    def _test_dataloader(self, *, shuffle: bool = True, drop_last: bool = True) -> DataLoader:
        ...

    @implements(pl.LightningDataModule)
    def train_dataloader(self, *, shuffle: bool = True, drop_last: bool = True) -> DataLoader:
        return self._train_dataloader(shuffle=shuffle, drop_last=drop_last)

    @implements(pl.LightningDataModule)
    def val_dataloader(self, *, shuffle: bool = False, drop_last: bool = False) -> DataLoader:
        return self._val_dataloader(shuffle=shuffle, drop_last=drop_last)

    @implements(pl.LightningDataModule)
    def test_dataloader(self, *, shuffle: bool = False, drop_last: bool = False) -> DataLoader:
        return self._test_dataloader(shuffle=shuffle, drop_last=drop_last)

    def scale_and_split(
        self,
        datatuple: DataTuple,
        dataset: Dataset,
    ) -> ScaleSplitOut:
        """Scale a datatuple and split to train/test."""

        num_train = int(datatuple.x.shape[0] * 0.7)
        num_val = int(datatuple.x.shape[0] * 0.1)
        rng = np.random.RandomState(self.seed)
        idx = rng.permutation(datatuple.x.index)
        if self.train_indices is None:
            self.train_indices = idx[:num_train]
        if self.val_indices is None:
            self.val_indices = idx[num_train : num_train + num_val]
        if self.test_indices is None:
            self.test_indices = idx[num_train + num_val :]

        train = DataTuple(
            x=datatuple.x.iloc[self.train_indices].reset_index(drop=True),  # type: ignore[union-attr, index]
            s=datatuple.s.iloc[self.train_indices].reset_index(drop=True),  # type: ignore[union-attr, index]
            y=datatuple.y.iloc[self.train_indices].reset_index(drop=True),  # type: ignore[union-attr, index]
        )
        val = DataTuple(
            x=datatuple.x.iloc[self.val_indices].reset_index(drop=True),  # type: ignore[union-attr, index]
            s=datatuple.s.iloc[self.val_indices].reset_index(drop=True),  # type: ignore[union-attr, index]
            y=datatuple.y.iloc[self.val_indices].reset_index(drop=True),  # type: ignore[union-attr, index]
        )
        test = DataTuple(
            x=datatuple.x.iloc[self.test_indices].reset_index(drop=True),  # type: ignore[union-attr, index]
            s=datatuple.s.iloc[self.test_indices].reset_index(drop=True),  # type: ignore[union-attr, index]
            y=datatuple.y.iloc[self.test_indices].reset_index(drop=True),  # type: ignore[union-attr, index]
        )

        if self.scaler is not None:
            self.scaler = self.scaler.fit(train.x[dataset.continuous_features])
            train.x[dataset.continuous_features] = self.scaler.transform(
                train.x[dataset.continuous_features]
            )
            val.x[dataset.continuous_features] = (
                self.scaler.transform(val.x[dataset.continuous_features])
                if val.x.shape[0] > 0
                else val.x[dataset.continuous_features]
            )
            test.x[dataset.continuous_features] = self.scaler.transform(
                test.x[dataset.continuous_features]
            )
        return ScaleSplitOut(train=train, val=val, test=test)

    def make_data_plots(self, *, cf_available: bool, logger: pll.WandbLogger) -> None:
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
                label_plot(self.cf_train_datatuple, logger, "cf_train")
                label_plot(self.cf_test_datatuple, logger, "cf_test")
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
