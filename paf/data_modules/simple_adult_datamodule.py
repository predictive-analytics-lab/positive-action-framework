"""Adult Dataset DataModule."""
from __future__ import annotations

from ethicml import ProportionalSplit, RandomSplit
from kit import implements
import numpy as np
from pytorch_lightning import LightningDataModule
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

from paf.base_templates.base_module import BaseDataModule
from paf.base_templates.dataset_utils import DataTupleDataset
from paf.datasets.ethicml_datasets import adult_data


class SimpleAdultDataModule(BaseDataModule):
    """Simple 1d, configurable, data."""

    def __init__(
        self,
        batch_size: int,
        bin_nat: bool,
        bin_race: bool,
        seed: int,
        num_workers: int,
        sens: str,
        cf_available: bool = False,
    ):
        super().__init__()
        self._cf_available = cf_available
        self.batch_size = batch_size
        self.bin_nat = bin_nat
        self.bin_race = bin_race
        self.seed = seed
        self.num_workers = num_workers
        self.sens = sens

    @implements(LightningDataModule)
    def prepare_data(self) -> None:
        self.dataset, self.factual_data = adult_data(
            sens=self.sens, bin_nationality=self.bin_nat, bin_race=self.bin_race
        )
        self.card_s = self.factual_data.s.nunique().values[0]
        self.data_dim = self.factual_data.x.shape[1:]
        self.dims = self.data_dim
        self.dim_s = (1,) if self.factual_data.s.ndim == 1 else self.factual_data.s.shape[1:]
        self.column_names = self.dataset.discrete_features + self.dataset.continuous_features
        self.outcome_columns = [str(col) for col in self.factual_data.y.columns]

        train_val_datatuple, self.test_datatuple, split_info = ProportionalSplit(
            train_percentage=0.8
        )(self.factual_data)

        self.train_datatuple, self.val_datatuple, _ = RandomSplit(train_percentage=0.875)(
            train_val_datatuple
        )

        self.scaler = MinMaxScaler()
        self.scaler = self.scaler.fit(self.train_datatuple.x[self.dataset.continuous_features])
        self.train_datatuple.x[self.dataset.continuous_features] = self.scaler.transform(
            self.train_datatuple.x[self.dataset.continuous_features]
        )
        self.test_datatuple.x[self.dataset.continuous_features] = self.scaler.transform(
            self.test_datatuple.x[self.dataset.continuous_features]
        )

        self.make_feature_groups(self.dataset, self.factual_data)

    @implements(BaseDataModule)
    def _train_dataloader(self, shuffle: bool = False, drop_last: bool = False) -> DataLoader:
        return DataLoader(
            DataTupleDataset(
                self.train_datatuple,
                disc_features=self.dataset.discrete_features,
                cont_features=self.dataset.continuous_features,
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
        )

    @implements(BaseDataModule)
    def _test_dataloader(self, shuffle: bool = False, drop_last: bool = False) -> DataLoader:
        return DataLoader(
            DataTupleDataset(
                self.test_datatuple,
                disc_features=self.dataset.discrete_features,
                cont_features=self.dataset.continuous_features,
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
        )
