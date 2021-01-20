"""Augmented dataset."""
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from torch.utils.data.dataset import T_co

from src.data_modules.base_module import BaseDataModule
from src.model.encoder_model import AE


class AugDataset(Dataset):
    """Aigmented Dataset."""

    def __init__(self, recons: Tensor, sens: Tensor, labels: Tensor):
        super().__init__()
        self.recons = recons
        self.sens = sens
        self.sens_0 = torch.zeros_like(self.sens)
        self.sens_1 = torch.ones_like(self.sens)
        self.labels = labels

    def __len__(self) -> int:
        return self.sens.shape[0]

    def __getitem__(self, index) -> T_co:
        return (
            (self.recons[0][index], self.sens[index], self.sens_0[index], self.labels[index]),
            (self.recons[1][index], self.sens[index], self.sens_1[index], self.labels[index]),
        )


class AugmentedDataModule(BaseDataModule):
    """Augmented Dataset."""

    def __init__(self, data: BaseDataModule, model: AE):
        super().__init__()
        self.recons, self.sens, self.labels = model.run_through(data.train_dataloader())

    @staticmethod
    def collate_tuples(batch):
        """Callate functin returning outpusts concatenated."""
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
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
        pass
