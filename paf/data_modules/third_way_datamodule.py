"""Data Module for simple data."""
from __future__ import annotations
import logging
from typing import Optional, Tuple

from ethicml import Dataset, DataTuple
import pandas as pd
import pytorch_lightning as pl
from ranzen import implements, parsable
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

from paf.base_templates.base_module import BaseDataModule
from paf.base_templates.dataset_utils import CFDataTupleDataset
from paf.datasets.third_way import third_way_data
from paf.selection import selection_rules
from paf.utils import facct_mapper

__all__ = ["ThirdWayDataModule"]

log = logging.getLogger(__name__)


class ThirdWayDataModule(BaseDataModule):
    """Simple 1d, configurable, data."""

    dataset: Dataset
    factual_data: DataTuple
    cf_data: DataTuple
    s1_0_s2_0_data: DataTuple
    s1_0_s2_1_data: DataTuple
    s1_1_s2_0_data: DataTuple
    s1_1_s2_1_data: DataTuple
    card_s: int
    card_y: int
    column_names: list[str]
    outcome_columns: list[str]
    val_datatuple: DataTuple
    train_datatuple: DataTuple
    true_val_datatuple: DataTuple
    true_test_datatuple: DataTuple
    cf_train_datatuple: DataTuple
    cf_val_datatuple: DataTuple
    cf_test_datatuple: DataTuple

    @parsable
    def __init__(
        self,
        acceptance_rate: float,
        alpha: float,
        beta: float,
        gamma: float,
        seed: int,
        num_samples: int,
        num_workers: int,
        batch_size: int,
        num_features: int,
        xi: float,
        num_hidden_features: int,
        cf_available: bool = True,
        train_dims: Optional[Tuple[int, ...]] = None,
    ):
        super().__init__(cf_available=cf_available, seed=seed, scaler=MinMaxScaler())
        self.acceptance_rate = acceptance_rate
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.num_samples = num_samples
        self.train_dims = train_dims
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.num_features = num_features
        self.xi = xi
        self.num_hidden_features = num_hidden_features

    @implements(pl.LightningDataModule)
    def prepare_data(self) -> None:
        # called only on 1 GPU
        data = third_way_data(
            seed=self.seed,
            num_samples=self.num_samples,
            acceptance_rate=self.acceptance_rate,
            alpha=self.alpha,
            gamma=self.gamma,
            random_shift=0,
            binary_s=1,
            num_features=self.num_features,
            beta=self.beta,
            xi=self.xi,
            num_hidden_features=self.num_hidden_features,
        )

        dts = self.scale_and_split(data.data, data.dataset)
        true_dts = self.scale_and_split(data.data_true_outcome, data.dataset)
        cf_dts = self.scale_and_split(data.cf_data, data.dataset)

        self.set_data_values(
            dataset=data.dataset,
            best_guess=data.cf_groups,
            factual_data=data.data,
            true_dts=true_dts,
            cf_dts=cf_dts,
            dts=dts,
        )

        s10s20_dts = self.scale_and_split(data.data_xs0_ys0, data.dataset)
        s1_0_s2_0_test = s10s20_dts.test
        s10s21_dts = self.scale_and_split(data.data_xs0_ys1, data.dataset)
        s1_0_s2_1_test = s10s21_dts.test
        s11s20_dts = self.scale_and_split(data.data_xs1_ys0, data.dataset)
        s1_1_s2_0_test = s11s20_dts.test
        s11s21_dts = self.scale_and_split(data.data_xs1_ys1, data.dataset)
        s1_1_s2_1_test = s11s21_dts.test

        pd_results = pd.concat(
            [
                s1_0_s2_0_test.y.rename(columns={"outcome": "s1_0_s2_0"}),
                s1_0_s2_1_test.y.rename(columns={"outcome": "s1_0_s2_1"}),
                s1_1_s2_0_test.y.rename(columns={"outcome": "s1_1_s2_0"}),
                s1_1_s2_1_test.y.rename(columns={"outcome": "s1_1_s2_1"}),
                self.data_group.test.s.rename(columns={"sens": "true_s"}),
                self.data_group.test.y.rename(columns={"outcome": "actual"}),
            ],
            axis=1,
        )
        log.info(pd_results.info)

        pd_results["decision"] = selection_rules(pd_results)
        log.info(pd_results["decision"].value_counts())
        log.info(facct_mapper(pd.Series(pd_results["decision"])))

    @implements(BaseDataModule)
    def _train_dataloader(self, *, shuffle: bool = False, drop_last: bool = False) -> DataLoader:
        return DataLoader(
            CFDataTupleDataset(
                self.train_datatuple,
                cf_dataset=self.cf_train_datatuple,
                disc_features=self.dataset.discrete_features,
                cont_features=self.dataset.continuous_features,
            ),
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
        )

    @implements(BaseDataModule)
    def _val_dataloader(self, *, shuffle: bool = False, drop_last: bool = False) -> DataLoader:
        return DataLoader(
            CFDataTupleDataset(
                dataset=self.val_datatuple,
                cf_dataset=self.cf_val_datatuple,
                cont_features=self.dataset.continuous_features,
                disc_features=self.dataset.discrete_features,
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=drop_last,
            shuffle=shuffle,
        )

    @implements(BaseDataModule)
    def _test_dataloader(self, *, shuffle: bool = False, drop_last: bool = False) -> DataLoader:
        return DataLoader(
            CFDataTupleDataset(
                self.data_group.test,
                cf_dataset=self.cf_test_datatuple,
                disc_features=self.dataset.discrete_features,
                cont_features=self.dataset.continuous_features,
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
        )
