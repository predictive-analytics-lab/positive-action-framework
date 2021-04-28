"""Data Module for simple data."""
from typing import Optional, Tuple

import numpy as np
from ethicml import implements
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from paf.base_templates.base_module import BaseDataModule
from paf.base_templates.dataset_utils import CFDataTupleDataset
from paf.datasets.lilliput import lilliput


class LilliputDataModule(BaseDataModule):
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
        (
            dataset,
            factual_data,
            cf_data,
            data_true_outcome,
            best_guess,
            s0s0,
            s0s1,
            s1s0,
            s1s1,
        ) = lilliput(
            seed=self.seed, alpha=self.alpha, num_samples=self.num_samples, gamma=self.gamma
        )
        self.best_guess = best_guess
        self.s0_s0 = s0s0
        self.s0_s1 = s0s1
        self.s1_s0 = s1s0
        self.s1_s1 = s1s1
        self.dataset = dataset
        self.factual_data = factual_data
        self.cf_data = cf_data
        self.num_s = factual_data.s.nunique().values[0]
        self.data_dim = factual_data.x.shape[1]
        self.s_dim = factual_data.s.shape[1]
        self.column_names = factual_data.x.columns
        self.outcome_columns = factual_data.y.columns

        num_train = int(self.factual_data.x.shape[0] * 0.8)
        num_val = 0  # int(self.factual_data.x.shape[0] * 0.1)
        rng = np.random.RandomState(self.seed)
        idx = rng.permutation(self.factual_data.x.index)
        train_indices = idx[:num_train]
        val_indices = idx[num_train : num_train + num_val]
        test_indices = idx[num_train + num_val :]

        self.train_data, self.val_data, self.test_data = self.scale_and_split(
            self.factual_data, dataset, train_indices, val_indices, test_indices
        )
        self.true_train_data, self.true_val_data, self.true_test_data = self.scale_and_split(
            data_true_outcome, dataset, train_indices, val_indices, test_indices
        )
        self.cf_train, self.cf_val, self.cf_test = self.scale_and_split(
            self.cf_data, dataset, train_indices, val_indices, test_indices
        )

        self.make_feature_groups(dataset, factual_data)

    @implements(BaseDataModule)
    def _train_dataloader(self, shuffle: bool = True, drop_last: bool = True) -> DataLoader:
        return DataLoader(
            CFDataTupleDataset(
                dataset=self.train_data,
                cf_dataset=self.cf_train,
                disc_features=self.dataset.discrete_features,
                cont_features=self.dataset.continuous_features,
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
        )

    @implements(LightningDataModule)
    def val_dataloader(self, shuffle: bool = False, drop_last: bool = False) -> DataLoader:
        return DataLoader(
            CFDataTupleDataset(
                dataset=self.val_data,
                cf_dataset=self.cf_val,
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
                dataset=self.test_data,
                cf_dataset=self.cf_test,
                disc_features=self.dataset.discrete_features,
                cont_features=self.dataset.continuous_features,
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
        )
