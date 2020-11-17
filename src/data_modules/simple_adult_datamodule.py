"""Adult Dataset DataModule."""
from ethicml import implements, train_test_split
from pytorch_lightning import LightningDataModule
from sklearn.preprocessing import MinMaxScaler
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
        self._num_s = -1
        self._s_dim = -1
        self._x_dim = -1
        self.batch_size = cfg.batch_size
        self.bin_nat = cfg.bin_nationality
        self.seed = cfg.seed
        self.num_workers = cfg.num_workers

    @implements(LightningDataModule)
    def prepare_data(self) -> None:
        dataset, true_data = adult_data(bin_nationality=self.bin_nat)
        self.dataset = dataset
        self.num_s = true_data.s.nunique().values[0]
        self.data_dim = true_data.x.shape[1]
        self.s_dim = true_data.s.shape[1]

        train, test = train_test_split(true_data, 0.8, self.seed)

        scaler = MinMaxScaler()
        scaler = scaler.fit(train.x[dataset.cont_features])
        train.x[dataset.cont_features] = scaler.transform(train.x[dataset.cont_features])
        test.x[dataset.cont_features] = scaler.transform(test.x[dataset.cont_features])

        self.train_data = train
        self.test_data = test
        self.make_feature_groups(dataset, true_data)

    @implements(LightningDataModule)
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            DataTupleDataset(
                self.train_data,
                disc_features=self.dataset.discrete_features,
                cont_features=self.dataset.cont_features,
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    @implements(LightningDataModule)
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            DataTupleDataset(
                self.test_data,
                disc_features=self.dataset.discrete_features,
                cont_features=self.dataset.cont_features,
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
