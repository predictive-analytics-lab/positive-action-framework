"""Data Module for simple data."""
from typing import Tuple

import numpy as np
from ethicml import Dataset, DataTuple, implements
from pytorch_lightning import LightningDataModule
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

from src.config_classes.dataclasses import LilliputConfig
from src.data_modules.base_module import BaseDataModule
from src.data_modules.dataset_utils import CFDataTupleDataset
from src.datasets.lilliput import lilliput


class LilliputDataModule(BaseDataModule):
    """Simple 1d, configurable, data."""

    def __init__(self, cfg: LilliputConfig):
        super().__init__()
        self.alpha = cfg.alpha
        self._cf_available = True
        self.gamma = cfg.gamma
        self.seed = cfg.seed
        self.num_samples = cfg.num_samples
        self.train_dims = None
        self.num_workers = cfg.num_workers
        self.batch_size = cfg.batch_size

    @implements(LightningDataModule)
    def prepare_data(self) -> None:
        # called only on 1 GPU
        dataset, factual_data, cf_data, data_true_outcome, best_guess = lilliput(
            seed=self.seed, alpha=self.alpha, num_samples=self.num_samples
        )
        self.best_guess = best_guess
        self.dataset = dataset
        self.factual_data = factual_data
        self.cf_data = cf_data
        self.num_s = factual_data.s.nunique().values[0]
        self.data_dim = factual_data.x.shape[1]
        self.s_dim = factual_data.s.shape[1]
        self.column_names = factual_data.x.columns
        self.outcome_columns = factual_data.y.columns

        num_train = int(self.factual_data.x.shape[0] * 0.8)
        rng = np.random.RandomState(self.seed)
        idx = rng.permutation(self.factual_data.x.index)
        train_indices = idx[:num_train]
        test_indices = idx[num_train:]

        self.train_data, self.test_data = self.scale_and_split(self.factual_data, dataset, train_indices, test_indices)
        self.true_train_data, self.true_test_data = self.scale_and_split(
            data_true_outcome, dataset, train_indices, test_indices
        )
        self.cf_train, self.cf_test = self.scale_and_split(self.cf_data, dataset, train_indices, test_indices)

        self.make_feature_groups(dataset, factual_data)

    def scale_and_split(
        self, datatuple: DataTuple, dataset: Dataset, train_indices: np.ndarray, test_indices: np.ndarray
    ) -> Tuple[DataTuple, DataTuple]:
        """Scale a datatuple and split to train/test."""
        train = DataTuple(
            x=datatuple.x.iloc[train_indices].reset_index(drop=True),
            s=datatuple.s.iloc[train_indices].reset_index(drop=True),
            y=datatuple.y.iloc[train_indices].reset_index(drop=True),
        )
        test = DataTuple(
            x=datatuple.x.iloc[test_indices].reset_index(drop=True),
            s=datatuple.s.iloc[test_indices].reset_index(drop=True),
            y=datatuple.y.iloc[test_indices].reset_index(drop=True),
        )

        scaler = MinMaxScaler()
        scaler = scaler.fit(train.x[dataset.continuous_features])
        train.x[dataset.continuous_features] = scaler.transform(train.x[dataset.continuous_features])
        test.x[dataset.continuous_features] = scaler.transform(test.x[dataset.continuous_features])
        return train, test

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
