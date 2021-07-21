"""Augmented dataset."""
from typing import List

from ethicml import DataTuple, compute_instance_weights
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from torch.utils.data.dataset import T_co

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

    def __getitem__(self, index: int) -> T_co:
        return (
            (
                self.recons[0][index],
                self.sens[index],
                self.sens_0[index],
                self.labels[index],
                self.instance_weight[index],
            ),
            (
                self.recons[1][index],
                self.sens[index],
                self.sens_1[index],
                self.labels[index],
                self.instance_weight[index],
            ),
        )


class AugmentedDataModule(BaseDataModule):
    """Augmented Dataset."""

    def __init__(self, data: BaseDataModule, model: AE):
        super().__init__()
        self.recons, self.sens, self.labels = model.run_through(data.train_dataloader())

    @staticmethod
    def collate_tuples(batch: Tensor) -> List[Tensor]:
        """Callate functin returning outpusts concatenated."""
        it = iter(batch)
        elem_size = len(next(it))
        if any(len(elem) != elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        collated = [default_collate(samples) for samples in transposed]
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
