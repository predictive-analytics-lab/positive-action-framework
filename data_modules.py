from ethicml import DataTuple
from pytorch_lightning import LightningDataModule
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

from config_classes.dataclasses import DataConfig
from dataset_utils import DataTupleDataset
from simple_x import simple_x_data
import numpy as np


class SimpleXDataModule(LightningDataModule):
    def __init__(self, cfg: DataConfig):
        super().__init__()
        self.alpha = cfg.alpha
        self.gamma = cfg.gamma
        self.seed = cfg.seed
        self.num_samples = cfg.num_samples
        self.train_dims = None
        self.num_s = 2

    def prepare_data(self):
        # called only on 1 GPU
        dataset, true_data, cf_data = simple_x_data(
            seed=self.seed,
            num_samples=self.num_samples,
            alpha=self.alpha,
            gamma=self.gamma,
            random_shift=0,
            binary_s=1,
        )
        self.dataset = dataset
        self.true_data = true_data
        self.cf_data = cf_data
        self.num_s = true_data.s.nunique()

        num_train = int(self.true_data.x.shape[0] * 0.8)
        rng = np.random.RandomState(self.seed)
        idx = rng.permutation(self.true_data.x.index)
        train_indices = idx[:num_train]
        test_indices = idx[num_train:]

        train = DataTuple(
            x=self.true_data.x.iloc[train_indices].reset_index(drop=True),
            s=self.true_data.s.iloc[train_indices].reset_index(drop=True),
            y=self.true_data.y.iloc[train_indices].reset_index(drop=True),
        )
        test = DataTuple(
            x=self.true_data.x.iloc[test_indices].reset_index(drop=True),
            s=self.true_data.s.iloc[test_indices].reset_index(drop=True),
            y=self.true_data.y.iloc[test_indices].reset_index(drop=True),
        )

        scaler = MinMaxScaler()
        scaler = scaler.fit(train.x[dataset.cont_features])
        train.x[dataset.cont_features] = scaler.transform(train.x[dataset.cont_features])
        test.x[dataset.cont_features] = scaler.transform(test.x[dataset.cont_features])

        self.train_data = train
        self.test_data = test

        counterfactual_train = DataTuple(
            x=self.cf_data.x.iloc[train_indices].reset_index(drop=True),
            s=self.cf_data.s.iloc[train_indices].reset_index(drop=True),
            y=self.cf_data.y.iloc[train_indices].reset_index(drop=True),
        )
        counterfactual_test = DataTuple(
            x=self.cf_data.x.iloc[test_indices].reset_index(drop=True),
            s=self.cf_data.s.iloc[test_indices].reset_index(drop=True),
            y=self.cf_data.y.iloc[test_indices].reset_index(drop=True),
        )

        self.scaler_cf = MinMaxScaler()
        self.scaler_cf = self.scaler_cf.fit(counterfactual_train.x[dataset.cont_features])
        counterfactual_train.x[dataset.cont_features] = self.scaler_cf.transform(
            counterfactual_train.x[dataset.cont_features]
        )
        counterfactual_test.x[dataset.cont_features] = self.scaler_cf.transform(
            counterfactual_test.x[dataset.cont_features]
        )

        self.cf_train = counterfactual_train
        self.cf_test = counterfactual_test

    def train_dataloader(self):
        return DataLoader(
            DataTupleDataset(
                self.train_data,
                self.cf_train,
                disc_features=self.dataset.discrete_features,
                cont_features=self.dataset.cont_features,
            ),
            batch_size=64,
            num_workers=0,
            shuffle=True,
        )

    def test_dataloader(self):
        return DataLoader(
            DataTupleDataset(
                self.test_data,
                self.cf_test,
                disc_features=self.dataset.discrete_features,
                cont_features=self.dataset.cont_features,
            ),
            batch_size=64,
            num_workers=0,
            shuffle=False,
        )
