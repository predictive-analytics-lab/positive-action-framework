"""Adult Dataset DataModule."""
from __future__ import annotations

from kit import implements
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
        super().__init__(cf_available=cf_available, seed=seed)
        self.batch_size = batch_size
        self.bin_nat = bin_nat
        self.bin_race = bin_race
        self.num_workers = num_workers
        self.sens = sens

    @implements(LightningDataModule)
    def prepare_data(self) -> None:
        dataset, factual_data = adult_data(
            sens=self.sens, bin_nationality=self.bin_nat, bin_race=self.bin_race
        )

        self.set_data_values(
            dataset=dataset,
            dts=self.scale_and_split(factual_data, dataset),
            factual_data=factual_data,
            scaler=MinMaxScaler(),
        )

    @implements(BaseDataModule)
    def _train_dataloader(self, *, shuffle: bool = False, drop_last: bool = False) -> DataLoader:
        return DataLoader(
            DataTupleDataset(
                dataset=self.train_datatuple,
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
                dataset=self.val_datatuple,
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
                dataset=self.test_datatuple,
                disc_features=self.dataset.discrete_features,
                cont_features=self.dataset.continuous_features,
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
        )
