"""Adult Dataset DataModule."""
import numpy as np
from ethicml import implements
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.config_classes.dataclasses import AdultConfig
from src.data_modules.base_module import BaseDataModule
from src.data_modules.dataset_utils import DataTupleDataset
from src.datasets.adult_dataset import adult_data


class SimpleAdultDataModule(BaseDataModule):
    """Simple 1d, configurable, data."""

    def __init__(self, cfg: AdultConfig):
        super().__init__()
        self._cf_available = False
        self.batch_size = cfg.batch_size
        self.bin_nat = cfg.bin_nationality
        self.bin_race = cfg.bin_race
        self.seed = cfg.seed
        self.num_workers = cfg.num_workers
        self.sens = cfg.sens

    @implements(LightningDataModule)
    def prepare_data(self) -> None:
        self.dataset, self.factual_data = adult_data(
            sens=self.sens, bin_nationality=self.bin_nat, bin_race=self.bin_race
        )
        self.num_s = self.factual_data.s.nunique().values[0]
        self.data_dim = self.factual_data.x.shape[1]
        self.s_dim = self.factual_data.s.shape[1]
        self.column_names = self.dataset.discrete_features + self.dataset.continuous_features
        self.outcome_columns = self.factual_data.y.columns

        num_train = int(self.factual_data.x.shape[0] * 0.8)
        num_val = 0  # int(self.factual_data.x.shape[0] * 0.1)
        rng = np.random.RandomState(self.seed)
        idx = rng.permutation(self.factual_data.x.index)
        train_indices = idx[:num_train]
        val_indices = idx[num_train : num_train + num_val]
        test_indices = idx[num_train + num_val :]

        self.train_data, self.val_data, self.test_data = self.scale_and_split(
            self.factual_data, self.dataset, train_indices, val_indices, test_indices
        )

        self.make_feature_groups(self.dataset, self.factual_data)

    @implements(BaseDataModule)
    def _train_dataloader(self, shuffle: bool = False, drop_last: bool = False) -> DataLoader:
        return DataLoader(
            DataTupleDataset(
                self.train_data,
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
                self.test_data,
                disc_features=self.dataset.discrete_features,
                cont_features=self.dataset.continuous_features,
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
        )
