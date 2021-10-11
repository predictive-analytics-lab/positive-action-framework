"""Adult Dataset DataModule."""
from __future__ import annotations

import pytorch_lightning as pl
from ranzen import implements, parsable
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

from paf.base_templates.base_module import BaseDataModule
from paf.base_templates.dataset_utils import DataTupleDataset
from paf.datasets.ethicml_datasets import semi_adult_data

__all__ = ["SemiAdultDataModule"]


class SemiAdultDataModule(BaseDataModule):
    """Simple 1d, configurable, data."""

    @parsable
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
        super().__init__(cf_available=cf_available, seed=seed, scaler=MinMaxScaler())
        self.batch_size = batch_size
        self.bin_nat = bin_nat
        self.bin_race = bin_race
        self.num_workers = num_workers
        self.sens = sens

    @implements(pl.LightningDataModule)
    def prepare_data(self) -> None:
        dataset, factual_data = semi_adult_data()

        self.set_data_values(
            dataset=dataset,
            best_guess=None,
            factual_data=factual_data,
            dts=self.scale_and_split(factual_data, dataset),
            cf_dts=None,
            true_dts=None,
        )

    @implements(BaseDataModule)
    def _train_dataloader(self, *, shuffle: bool = False, drop_last: bool = False) -> DataLoader:
        return DataLoader(
            DataTupleDataset(
                dataset=self.data_group.train,
                disc_features=self.dataset.discrete_features,
                cont_features=self.dataset.continuous_features,
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
        )

    @implements(BaseDataModule)
    def _val_dataloader(self, *, shuffle: bool = False, drop_last: bool = False) -> DataLoader:
        return DataLoader(
            DataTupleDataset(
                dataset=self.data_group.val,
                disc_features=self.dataset.discrete_features,
                cont_features=self.dataset.continuous_features,
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
        )

    @implements(BaseDataModule)
    def _test_dataloader(self, *, shuffle: bool = False, drop_last: bool = False) -> DataLoader:
        return DataLoader(
            DataTupleDataset(
                dataset=self.data_group.test,
                disc_features=self.dataset.discrete_features,
                cont_features=self.dataset.continuous_features,
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
        )
