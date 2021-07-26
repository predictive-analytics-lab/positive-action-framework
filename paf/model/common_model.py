"""Common methods for models."""
from abc import abstractmethod
from typing import Dict, List

import numpy as np
from pytorch_lightning import LightningModule
import torch
from torch.utils.data import DataLoader


class CommonModel(LightningModule):
    """Base Model for each component."""

    def __init__(self, name: str) -> None:
        super().__init__()
        self.model_name = name

    def get_latent(self, dataloader: DataLoader) -> np.ndarray:
        """Get Latents to be used post train/test."""
        latent = None
        for batch in dataloader:
            x = batch.x.to(self.device)
            s = batch.s.to(self.device)
            z, _, _ = self(x, s)
            latent = z if latent is None else torch.cat([latent, z], dim=0)
        assert latent is not None
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
        feature_groups: Dict[str, List[slice]],
        outcome_cols: List[str],
    ) -> None:
        """Build the network using data not available in advance."""
