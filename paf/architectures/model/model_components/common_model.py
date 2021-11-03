"""Common methods for models."""
from __future__ import annotations
from abc import abstractmethod

from conduit.fair.data import EthicMlDataModule
import numpy as np
import pytorch_lightning as pl
from ranzen import implements
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

__all__ = ["CommonModel", "BaseModel", "Encoder", "Adversary", "Decoder"]

from paf.base_templates import BaseDataModule

from .blocks import block, mid_blocks
from .model_utils import grad_reverse, to_discrete


class CommonModel(pl.LightningModule):
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
            z, _, _ = self.forward(x=x, s=s)
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
        *,
        num_s: int,
        data_dim: int,
        s_dim: int,
        cf_available: bool,
        feature_groups: dict[str, list[slice]],
        outcome_cols: list[str],
        data: BaseDataModule | EthicMlDataModule,
        indices: list[str],
    ) -> None:
        """Build the network using data not available in advance."""

    @torch.no_grad()
    def invert(self, z: Tensor, x: Tensor) -> Tensor:
        """Go from soft to discrete features."""
        k = z.detach().clone()
        if self.loss.feature_groups["discrete"]:
            for i in range(
                k[:, slice(self.loss.feature_groups["discrete"][-1].stop, k.shape[1])].shape[1]
            ):
                if i in []:  # [0]: Features to transplant to the reconstrcution
                    k[:, slice(self.loss.feature_groups["discrete"][-1].stop, k.shape[1])][
                        :, i
                    ] = x[:, slice(self.loss.feature_groups["discrete"][-1].stop, x.shape[1])][:, i]
                else:
                    k[:, slice(self.loss.feature_groups["discrete"][-1].stop, k.shape[1])][
                        :, i
                    ] = k[:, slice(self.loss.feature_groups["discrete"][-1].stop, k.shape[1])][
                        :, i
                    ]  # .sigmoid()
            for i, group_slice in enumerate(self.loss.feature_groups["discrete"]):
                if i in []:  # [2, 4]: Features to transplant
                    k[:, group_slice] = x[:, group_slice]
                else:
                    one_hot = to_discrete(inputs=k[:, group_slice])
                    k[:, group_slice] = one_hot
        # else:
        #     k = k  # .sigmoid()

        return k


class BaseModel(nn.Module):
    """Base AE Model."""

    hid: nn.Module
    out: nn.Module

    def __init__(self, *, in_size: int, hid_size: int, out_size: int, blocks: int):
        super().__init__()
        if blocks == 0:
            self.hid = nn.Identity()
            self.out = nn.Linear(in_size, out_size)
        else:
            _blocks = [block(in_dim=in_size, out_dim=hid_size)] + mid_blocks(
                latent_dim=hid_size, blocks=blocks
            )
            self.hid = nn.Sequential(*_blocks)
            self.out = nn.Linear(hid_size, out_size)
        nn.init.xavier_uniform_(self.out.weight)

    @implements(nn.Module)
    def forward(self, input_: Tensor) -> Tensor:
        hidden = self.hid(input_)
        return self.out(hidden)


class Encoder(BaseModel):
    """AE Shared Encoder."""

    def __init__(self, *, in_size: int, latent_dim: int, blocks: int, hid_multiplier: int):
        super().__init__(
            in_size=in_size,
            hid_size=latent_dim * hid_multiplier,
            out_size=latent_dim,
            blocks=blocks,
        )


class Adversary(BaseModel):
    """AE Adversary head."""

    def __init__(
        self,
        *,
        latent_dim: int,
        out_size: int,
        blocks: int,
        hid_multiplier: int,
        weight: float = 1.0,
    ):
        super().__init__(
            in_size=latent_dim,
            hid_size=latent_dim * hid_multiplier,
            out_size=out_size,
            blocks=blocks,
        )
        self.weight = weight

    @implements(nn.Module)
    def forward(self, input_: Tensor) -> Tensor:
        z_rev = grad_reverse(input_, lambda_=self.weight)
        return super().forward(z_rev)


class Decoder(BaseModel):
    """Decoder."""

    def __init__(self, *, latent_dim: int, in_size: int, blocks: int, hid_multiplier: int) -> None:
        super().__init__(
            in_size=latent_dim,
            hid_size=latent_dim * hid_multiplier,
            out_size=in_size,
            blocks=blocks,
        )
