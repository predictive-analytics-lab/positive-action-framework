"""Adult Dataset DataModule."""
import numpy as np
from ethicml import DataTuple, ProportionalSplit, implements
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

        # self.train_data, self.val_data, self.test_data = self.scale_and_split(
        #     self.factual_data, self.dataset, train_indices, val_indices, test_indices
        # )

        self.train_data, self.test_data, split_info = ProportionalSplit(train_percentage=0.8)(
            self.factual_data
        )  # BalancedTestSplit(train_percentage=0.8)(self.factual_data)

        self.val = DataTuple(
            x=self.train_data.x.iloc[val_indices].reset_index(drop=True),
            s=self.train_data.s.iloc[val_indices].reset_index(drop=True),
            y=self.train_data.y.iloc[val_indices].reset_index(drop=True),
        )

        scaler = MinMaxScaler()
        scaler = scaler.fit(self.train_data.x[self.dataset.continuous_features])
        self.train_data.x[self.dataset.continuous_features] = scaler.transform(
            self.train_data.x[self.dataset.continuous_features]
        )
        self.test_data.x[self.dataset.continuous_features] = scaler.transform(
            self.test_data.x[self.dataset.continuous_features]
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
