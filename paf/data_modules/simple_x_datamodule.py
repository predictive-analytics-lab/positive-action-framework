"""Data Module for simple data."""
from typing import Optional, Tuple

import numpy as np
from ethicml import implements
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from paf.data_modules.base_module import BaseDataModule
from paf.data_modules.dataset_utils import CFDataTupleDataset
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
        super().__init__()
        self.alpha = alpha
        self._cf_available = cf_available
        self.gamma = gamma
        self.seed = seed
        self.num_samples = num_samples
        self.train_dims = train_dims
        self.num_workers = num_workers
        self.batch_size = batch_size

    @implements(LightningDataModule)
    def prepare_data(self) -> None:
        # called only on 1 GPU
        dataset, true_data, cf_data, data_true_outcome = simple_x_data(
            seed=self.seed,
            num_samples=self.num_samples,
            alpha=self.alpha,
            gamma=self.gamma,
            random_shift=0,
            binary_s=1,
        )
        self.dataset = dataset
        self.num_s = true_data.s.nunique().values[0]
        self.data_dim = true_data.x.shape[1]
        self.s_dim = true_data.s.shape[1]
        self.column_names = true_data.x.columns
        self.outcome_columns = true_data.y.columns

        num_train = int(true_data.x.shape[0] * 0.8)
        rng = np.random.RandomState(self.seed)
        idx = rng.permutation(true_data.x.index)
        train_indices = idx[:num_train]
        test_indices = idx[num_train:]

        self.make_feature_groups(dataset, true_data)

        self.train_data, self.test_data = self.scale_and_split(
            true_data, dataset, train_indices, test_indices
        )
        self.true_train_data, self.true_test_data = self.scale_and_split(
            data_true_outcome, dataset, train_indices, test_indices
        )
        self.cf_train, self.cf_test = self.scale_and_split(
            cf_data, dataset, train_indices, test_indices
        )

    @implements(BaseDataModule)
    def _train_dataloader(self, shuffle: bool = True, drop_last: bool = True) -> DataLoader:
        return DataLoader(
            CFDataTupleDataset(
                self.train_data,
                cf_dataset=self.cf_train,
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
            CFDataTupleDataset(
                self.test_data,
                cf_dataset=self.cf_test,
                disc_features=self.dataset.discrete_features,
                cont_features=self.dataset.continuous_features,
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
        )
