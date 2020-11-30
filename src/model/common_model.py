"""Common methods for models."""
from abc import abstractmethod

import numpy as np
import torch
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader


class CommonModel(LightningModule):
    """Base Model for each component."""

    def __init__(self) -> None:
        super().__init__()

    def get_latent(self, dataloader: DataLoader) -> np.ndarray:
        """Get Latents to be used post train/test."""
        latent = None
        for batch in dataloader:
            if self.cf_model:
                x, s, y, cf_x, cf_s, cf_y = batch
            else:
                x, s, y = batch
            x = x.to(self.device)
            s = s.to(self.device)
            z, _, _ = self(x, s)
            latent = z if latent is None else torch.cat([latent, z], dim=0)  # type: ignore[unreachable]
        assert latent is not None
        return latent.detach().cpu().numpy()

    @abstractmethod
    def get_recon(self, dataloader: DataLoader) -> np.ndarray:
        """Get Reconstructions to be used post train/test."""
