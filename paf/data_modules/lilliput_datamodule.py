"""Data Module for simple data."""
from __future__ import annotations
from typing import Optional, Tuple

from ethicml import DataTuple
from kit import implements, parsable
from pytorch_lightning import LightningDataModule
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

from paf.base_templates.base_module import BaseDataModule
from paf.base_templates.dataset_utils import CFDataTupleDataset
from paf.datasets.lilliput import lilliput


class LilliputDataModule(BaseDataModule):
    """Simple 1d, configurable, data."""

    cf_train_datatuple: DataTuple
    cf_val_datatuple: DataTuple
    cf_test_datatuple: DataTuple
    true_train_datatuple: DataTuple
    true_val_datatuple: DataTuple
    true_test_datatuple: DataTuple
    train_datatuple: DataTuple
    val_datatuple: DataTuple
    test_datatuple: DataTuple
    dim_x: tuple[int, ...]
    card_y: int
    cf_data: DataTuple
    factual_data: DataTuple
    s0_s0: DataTuple
    s0_s1: DataTuple
    s1_s0: DataTuple
    s1_s1: DataTuple
    column_names: list[str]
    outcome_columns: list[str]

    @parsable
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
        cf_data = lilliput(
            seed=self.seed, alpha=self.alpha, num_samples=self.num_samples, gamma=self.gamma
        )
        self.s0_s0 = cf_data.data_xs0_ys0
        self.s0_s1 = cf_data.data_xs0_ys1
        self.s1_s0 = cf_data.data_xs1_ys0
        self.s1_s1 = cf_data.data_xs1_ys1

        dts = self.scale_and_split(cf_data.data, cf_data.dataset)
        true_dts = self.scale_and_split(cf_data.data_true_outcome, cf_data.dataset)
        cf_dts = self.scale_and_split(cf_data.cf_data, cf_data.dataset)

        self.set_data_values(
            dataset=cf_data.dataset,
            cf_dts=cf_dts,
            true_dts=true_dts,
            dts=dts,
            factual_data=cf_data.data,
            best_guess=cf_data.cf_groups,
            scaler=MinMaxScaler(),
        )

    @implements(BaseDataModule)
    def _train_dataloader(self, *, shuffle: bool = True, drop_last: bool = True) -> DataLoader:
        return DataLoader(
            CFDataTupleDataset(
                dataset=self.train_datatuple,
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
        return DataLoader(
            CFDataTupleDataset(
                dataset=self.val_datatuple,
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
        return DataLoader(
            CFDataTupleDataset(
                dataset=self.test_datatuple,
                cf_dataset=self.cf_test_datatuple,
                disc_features=self.dataset.discrete_features,
                cont_features=self.dataset.continuous_features,
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
        )
