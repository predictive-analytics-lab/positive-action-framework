"""Data Module for simple data."""
from __future__ import annotations
from typing import Optional, Tuple

import pytorch_lightning as pl
from ranzen import implements, parsable
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

from paf.base_templates.base_module import BaseDataModule, CfOutcomes
from paf.base_templates.dataset_utils import CFDataTupleDataset
from paf.datasets.lilliput import lilliput

__all__ = ["LilliputDataModule"]


class LilliputDataModule(BaseDataModule):
    """Simple 1d, configurable, data."""

    @parsable
    def __init__(
        self,
        alpha: float,
        gamma: float,
        seed: int,
        num_samples: int,
        num_workers: int,
        train_batch_size: int,
        eval_batch_size: int,
        cf_available: bool = True,
        train_dims: Optional[Tuple[int, ...]] = None,
    ):
        super().__init__(
            cf_available=cf_available,
            seed=seed,
            scaler=MinMaxScaler(clip=True),
        )
        self.alpha = alpha
        self.gamma = gamma
        self.num_samples = num_samples
        self.train_dims = train_dims
        self.num_workers = num_workers
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

    @implements(pl.LightningDataModule)
    def prepare_data(self) -> None:
        # called only on 1 GPU
        cf_data = lilliput(
            seed=self.seed, alpha=self.alpha, num_samples=self.num_samples, gamma=self.gamma
        )

        dts = self.scale_and_split(cf_data.data, cf_data.dataset)
        true_dts = self.scale_and_split(cf_data.data_true_outcome, cf_data.dataset)
        cf_dts = self.scale_and_split(cf_data.cf_data, cf_data.dataset)

        cf_outcomes = CfOutcomes(
            s0_s0=cf_data.data_xs0_ys0,
            s0_s1=cf_data.data_xs0_ys1,
            s1_s0=cf_data.data_xs1_ys0,
            s1_s1=cf_data.data_xs1_ys1,
        )

        self.set_data_values(
            dataset=cf_data.dataset,
            cf_dts=cf_dts,
            true_dts=true_dts,
            dts=dts,
            factual_data=cf_data.data,
            best_guess=cf_data.cf_groups,
            cf_outcomes=cf_outcomes,
        )

    @implements(BaseDataModule)
    def _train_dataloader(self, *, shuffle: bool = True, drop_last: bool = True) -> DataLoader:
        assert self.cf_data_group is not None
        return DataLoader(
            CFDataTupleDataset(
                dataset=self.data_group.train,
                cf_dataset=self.cf_data_group.train,
                disc_features=self.dataset.discrete_features,
                cont_features=self.dataset.continuous_features,
            ),
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
        )

    @implements(BaseDataModule)
    def _val_dataloader(self, *, shuffle: bool = False, drop_last: bool = False) -> DataLoader:
        assert self.cf_data_group is not None
        return DataLoader(
            CFDataTupleDataset(
                dataset=self.data_group.val,
                cf_dataset=self.cf_data_group.val,
                disc_features=self.dataset.discrete_features,
                cont_features=self.dataset.continuous_features,
            ),
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
        )

    @implements(BaseDataModule)
    def _test_dataloader(self, *, shuffle: bool = False, drop_last: bool = False) -> DataLoader:
        assert self.cf_data_group is not None
        return DataLoader(
            CFDataTupleDataset(
                dataset=self.data_group.test,
                cf_dataset=self.cf_data_group.test,
                disc_features=self.dataset.discrete_features,
                cont_features=self.dataset.continuous_features,
            ),
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
        )
