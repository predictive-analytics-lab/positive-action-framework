"""Data Module for simple data."""
from __future__ import annotations
from typing import Optional, Tuple

from kit import implements
from pytorch_lightning import LightningDataModule
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

from paf.base_templates.base_module import BaseDataModule
from paf.base_templates.dataset_utils import CFDataTupleDataset
from paf.datasets.simple_x import simple_x_data


class SimpleXDataModule(BaseDataModule):
    """Simple 1d, configurable, data."""

    def __init__(
        self,
        alpha: float,
        gamma: float,
        seed: int,
        num_samples: int,
        num_workers: int,
        batch_size: int,
        cf_available: bool = True,
        train_dims: Optional[Tuple[int, ...]] = None,
    ):
        super().__init__(cf_available=cf_available, seed=seed)
        self.alpha = alpha
        self.gamma = gamma
        self.num_samples = num_samples
        self.train_dims = train_dims
        self.num_workers = num_workers
        self.batch_size = batch_size

    @implements(LightningDataModule)
    def prepare_data(self) -> None:
        # called only on 1 GPU
        data = simple_x_data(
            seed=self.seed,
            num_samples=self.num_samples,
            alpha=self.alpha,
            gamma=self.gamma,
            random_shift=0,
            binary_s=1,
        )

        dts = self.scale_and_split(data.true, data.dataset)
        true_dts = self.scale_and_split(data.true_outcomes, data.dataset)
        cf_dts = self.scale_and_split(data.cf, data.dataset)

        self.set_data_values(
            dataset=data.dataset,
            best_guess=None,
            factual_data=data.true,
            dts=dts,
            true_dts=true_dts,
            cf_dts=cf_dts,
            scaler=MinMaxScaler(),
        )

    @implements(BaseDataModule)
    def _train_dataloader(self, *, shuffle: bool = True, drop_last: bool = True) -> DataLoader:
        assert self.cf_train_datatuple is not None
        return DataLoader(
            CFDataTupleDataset(
                self.train_datatuple,
                cf_dataset=self.cf_train_datatuple,
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
        assert self.cf_val_datatuple is not None
        return DataLoader(
            CFDataTupleDataset(
                self.val_datatuple,
                cf_dataset=self.cf_val_datatuple,
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
        assert self.cf_test_datatuple is not None
        return DataLoader(
            CFDataTupleDataset(
                self.test_datatuple,
                cf_dataset=self.cf_test_datatuple,
                disc_features=self.dataset.discrete_features,
                cont_features=self.dataset.continuous_features,
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
        )
