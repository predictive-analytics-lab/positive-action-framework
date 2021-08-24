"""Augmented dataset."""
from __future__ import annotations
from typing import NamedTuple

from ethicml import DataTuple, compute_instance_weights
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate

from paf.base_templates.base_module import BaseDataModule
from paf.model.encoder_model import AE


class AugDataset(Dataset):
    """Aigmented Dataset."""

    def __init__(self, recons: Tensor, sens: Tensor, labels: Tensor):
        super().__init__()
        self.recons = recons
        self.sens = sens
        self.sens_0 = torch.zeros_like(self.sens)
        self.sens_1 = torch.ones_like(self.sens)
        self.labels = labels
        dataset = DataTuple(
            x=pd.DataFrame(recons[0].detach().cpu().numpy()),
            s=pd.DataFrame(sens.detach().cpu().numpy()),
            y=pd.DataFrame(labels.detach().cpu().numpy()),
        )
        self.instance_weight = torch.tensor(
            compute_instance_weights(dataset)["instance weights"].values
        )

    def __len__(self) -> int:
        return self.sens.shape[0]

    def __getitem__(self, index: int) -> AugBatch:
        return AugBatch(
            s0=AugItem(
                x=self.recons[0][index],
                s=self.sens[index],
                s0=self.sens_0[index],
                y=self.labels[index],
                iw=self.instance_weight[index],
            ),
            s1=AugItem(
                x=self.recons[1][index],
                s=self.sens[index],
                s0=self.sens_1[index],
                y=self.labels[index],
                iw=self.instance_weight[index],
            ),
        )


class AugItem(NamedTuple):
    x: Tensor
    s: Tensor
    s0: Tensor
    y: Tensor
    iw: Tensor


class AugBatch(NamedTuple):
    s0: AugItem
    s1: AugItem


class AugmentedDataModule(BaseDataModule):
    """Augmented Dataset."""

    def __init__(self, data: BaseDataModule, model: AE):
        super().__init__()
        self.recons, self.sens, self.labels = model.run_through(data.train_dataloader())

    @staticmethod
    def collate_tuples(batch: list[Tensor]) -> list[Tensor]:
        """Callate functin returning outpusts concatenated."""
        it = iter(batch)
        elem_size = len(next(it))
        if any(len(elem) != elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        collated = [default_collate(list(samples)) for samples in transposed]
        return [torch.cat([a, b]) for a, b in zip(collated[0], collated[1])]

    def _train_dataloader(self, shuffle: bool = True, drop_last: bool = True) -> DataLoader:
        return DataLoader(
            AugDataset(self.recons, self.sens, self.labels),
            batch_size=256,
            num_workers=0,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=self.collate_tuples,
        )

    def _test_dataloader(self, shuffle: bool = True, drop_last: bool = True) -> DataLoader:
        """This is only here to appease super. It's never used."""
