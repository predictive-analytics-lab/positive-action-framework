"""Common methods for models."""
from __future__ import annotations
from abc import abstractmethod

import numpy as np
from pytorch_lightning import LightningModule
from sklearn.preprocessing import MinMaxScaler
import torch
from torch import Tensor
from torch.utils.data import DataLoader

__all__ = ["CommonModel"]


class CommonModel(LightningModule):
    """Base Model for each component."""

    def __init__(self, name: str) -> None:
        super().__init__()
        self.model_name = name

    @property
    def name(self) -> str:
        return self.model_name

    def get_latent(self, dataloader: DataLoader) -> np.ndarray:
        """Get Latents to be used post train/test."""
        latent: list[Tensor] | None = None
        for batch in dataloader:
            x = batch.x.to(self.device)
            s = batch.s.to(self.device)
            z, _, _ = self.forward(x, s)
            if latent is None:
                latent = [z]
            else:
                latent.append(z)
        assert latent is not None
        latent = torch.cat(latent, dim=0)
        return latent.detach().cpu().numpy()

    @abstractmethod
    def get_recon(self, dataloader: DataLoader) -> np.ndarray:
        """Get Reconstructions to be used post train/test."""

    @abstractmethod
    def build(
        self,
        num_s: int,
        data_dim: int,
        s_dim: int,
        cf_available: bool,
        feature_groups: dict[str, list[slice]],
        outcome_cols: list[str],
        scaler: MinMaxScaler | None,
    ) -> None:
        """Build the network using data not available in advance."""
