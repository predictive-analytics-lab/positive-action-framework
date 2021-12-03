from __future__ import annotations

from conduit.types import Stage
from ranzen import implements

if 1:
    import faiss  # noqa

from abc import abstractmethod
from dataclasses import dataclass
import math
from typing import Any, Literal, NamedTuple, overload

import attr
from conduit.data import TernarySample
from conduit.fair.data import EthicMlDataModule
import numpy as np
import numpy.typing as npt
import pandas as pd
import pytorch_lightning as pl
import pytorch_lightning.utilities.types as plut
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from paf.architectures.model.model_components import (
    CommonModel,
    augment_recons,
    index_by_s,
)
from paf.base_templates import BaseDataModule
from paf.base_templates.dataset_utils import Batch, CfBatch

__all__ = ["NearestNeighbour", "NnStepOut"]


def pnorm(
    tensor_a: Tensor,
    tensor_b: Tensor,
    *,
    p: float = 2,
    root: bool = True,
    dim: int = -1,
) -> Tensor:
    dists = (tensor_a - tensor_b).abs()
    if math.isinf(p):
        if p > 0:
            norm = dists.max(dim).values
        else:
            norm = dists.min(dim).values
    else:
        norm = (dists ** p).sum(dim)
        if root:
            norm = norm ** (1 / p)  # type: ignore
    return norm


class KnnOutput(NamedTuple):
    indices: Tensor | npt.NDArray[np.uint]
    distances: Tensor | npt.NDArray[np.floating]


@attr.define(kw_only=True, eq=False)
class Knn(nn.Module):
    k: int
    p: float = 2
    root: bool = False
    normalize: bool = False
    """
    Whether to Lp-normalize the vectors for pairwise-distance computation.
    .. note::
        When vectors u and v are normalized to unit length, the Euclidean distance betwen them
        is equal to :math:`\\|u - v\\|^2 = (1-\\cos(u, v))`, that is the Euclidean distance over the end-points
        of u and v is a proper metric which gives the same ordering as the Cosine distance for
        any comparison of vectors, and furthermore avoids the potentially expensive trigonometric
        operations required to yield a proper metric.
    """

    def __attrs_pre_init__(self) -> None:
        super().__init__()

    @abstractmethod
    def _build_index(self, d: int) -> faiss.IndexFlat:
        ...

    def _index_to_gpu(self, x: Tensor, index: faiss.IndexFlat) -> faiss.GpuIndexFlat:  # type: ignore
        # use a single GPU
        res = faiss.StandardGpuResources()  # type: ignore
        # make it a flat GPU index
        return faiss.index_cpu_to_gpu(res, x.device.index, index)  # type: ignore

    @overload
    def forward(
        self,
        x: Tensor,
        *,
        y: Tensor | None = ...,
        return_distances: Literal[False] = ...,
    ) -> Tensor:
        ...

    @overload
    def forward(
        self,
        x: Tensor,
        *,
        y: Tensor | None = ...,
        return_distances: Literal[True] = ...,
    ) -> KnnOutput:
        ...

    def forward(
        self,
        x: Tensor,
        *,
        y: Tensor | None = None,
        return_distances: bool = False,
    ) -> Tensor | KnnOutput:

        x_np = x.detach().cpu().numpy()
        if self.normalize:
            x = F.normalize(x, dim=1, p=self.p)

        if y is None:
            y = x
            y_np = x_np
        else:
            if self.normalize:
                y = F.normalize(y, dim=1, p=self.p)
            y_np = y.detach().cpu().numpy()

        index = self._build_index(d=x.size(1))
        if x.is_cuda or y.is_cuda:
            index = self._index_to_gpu(x=x, index=index)

        if not index.is_trained:
            index.train(x=x_np)  # type: ignore
        # add vectors to the index
        index.add(x=y_np)  # type: ignore
        # search for the nearest k neighbors for each data-point
        distances_np, indices_np = index.search(x=x_np, k=self.k)  # type: ignore
        # Convert back from numpy to torch
        indices = torch.as_tensor(indices_np, device=x.device)

        if return_distances:
            if x.requires_grad or y.requires_grad:
                distances = pnorm(x[:, None], y[indices, :], dim=-1, p=self.p, root=False)
            else:
                distances = torch.as_tensor(distances_np, device=x.device)

            # Take the root of the distances to 'complete' the norm
            if self.root and (not math.isinf(self.p)):
                distances = distances ** (1 / self.p)

            return KnnOutput(indices=indices, distances=distances)
        return indices


@dataclass
class NnStepOut:
    cf_x: Tensor
    x: Tensor
    s: Tensor
    recons_0: Tensor
    recons_1: Tensor


@dataclass
class NnFwd:
    x: list[Tensor]


@attr.define(kw_only=True, eq=False)
class KnnExact(Knn):
    def _build_index(self, d: int) -> faiss.IndexFlat:
        index = faiss.IndexFlat(d, faiss.METRIC_Lp)
        index.metric_arg = self.p
        return index


class NearestNeighbour(CommonModel):
    name = "NearestNeighbour"
    all_preds: Tensor
    all_cf_preds: Tensor
    all_cf_x: Tensor
    all_x: Tensor
    all_s: Tensor
    all_y: Tensor
    pd_results: pd.DataFrame

    def __init__(self) -> None:
        super().__init__(name="NN")

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
        indices: list[str] | None,
    ) -> None:
        _ = (num_s, data_dim, s_dim, cf_available, feature_groups, indices)
        self.data_cols = outcome_cols
        self.train_features = torch.as_tensor(data.train_datatuple.x.values, dtype=torch.float32)
        self.train_sens = torch.as_tensor(data.train_datatuple.s.values, dtype=torch.long)

        # self.train_features = nn.Parameter(
        #     F.normalize(self.train_features.detach(), dim=1, p=2), requires_grad=False
        # ).float()

    def forward(self, *, x: Tensor, s: Tensor) -> NnFwd:
        # x = F.normalize(x, dim=1, p=2)
        knn = KnnExact(k=1, normalize=False)

        features = torch.empty_like(x)
        for s_val in range(2):
            s_val = torch.tensor(s_val).to(s.device)
            mask = (self.train_sens != s_val).squeeze()
            mask_inds = mask.nonzero(as_tuple=False).squeeze()
            knn_inds = knn(x=x[(s_val == s).squeeze()], y=self.train_features[mask_inds])
            abs_indices = mask_inds[knn_inds].squeeze()
            features[(s_val == s).squeeze()] = self.train_features[abs_indices]

        _x = augment_recons(x=x, cf_x=features, s=s)
        return NnFwd(x=[index_by_s(_x, torch.zeros_like(s)), index_by_s(_x, torch.ones_like(s))])

        # _ = self.knn(x=self.train_features[self.train_sens == 0], y=x[s == 1].nonzero())
        # ^^ set of x_s=1
        # repeat and get another set x_s=0
        # sets dont have order
        # result_0 == x_0

        # for point, s_label in zip(x, s):
        # print(f"{point.device=}")
        # print(f"{self.train_features.device=}")
        # print(f"{self.train_sens.device=}")
        # print(f"{s_label.device=}")
        # sim = point @ self.train_features[(self.train_sens != s_label).squeeze(-1)].t()
        # features.append(
        #     self.train_features[(self.train_sens != s_label).squeeze(-1)][sim.argmax(-1)]
        # )
        # return NnFwd(x=augment_recons(x, torch.stack(features, dim=0), s))

    def training_step(self, *_: Any) -> plut.STEP_OUTPUT:
        ...

    @implements(pl.LightningModule)
    def test_step(self, batch: Batch | CfBatch | TernarySample, *_: Any) -> NnStepOut:
        return self.shared_step(batch, stage=Stage.test)

    @implements(pl.LightningModule)
    def validation_step(self, batch: Batch | CfBatch | TernarySample, *_: Any) -> NnStepOut:
        return self.shared_step(batch, stage=Stage.validate)

    def shared_step(self, batch: Batch | CfBatch | TernarySample, *, stage: Stage) -> NnStepOut:
        recon_list = self.forward(x=batch.x, s=batch.s)

        return NnStepOut(
            cf_x=index_by_s(recon_list.x, 1 - batch.s),
            x=index_by_s(recon_list.x, batch.s),
            s=batch.s,
            recons_0=recon_list.x[0],
            recons_1=recon_list.x[1],
        )

    @implements(pl.LightningModule)
    def test_epoch_end(self, outputs: list[NnStepOut]) -> None:
        self.shared_epoch_end(outputs, stage=Stage.test)

    @implements(pl.LightningModule)
    def validation_epoch_end(self, outputs: list[NnStepOut]) -> None:
        self.shared_epoch_end(outputs, stage=Stage.validate)

    def shared_epoch_end(self, output_results: list[NnStepOut], *, stage: Stage) -> None:
        self.all_x = torch.cat([_r.x for _r in output_results], 0)
        self.all_s = torch.cat([_r.s for _r in output_results], 0)
        self.all_recon = torch.cat([_r.x for _r in output_results], 0)
        self.all_cf_pred = torch.cat([_r.cf_x for _r in output_results], 0)

    def predict_step(
        self, batch: Batch | CfBatch | TernarySample, batch_idx: int, *_: Any
    ) -> NnStepOut | None:
        recon_list = self.forward(x=batch.x, s=batch.s)

        return NnStepOut(
            cf_x=index_by_s(recon_list.x, 1 - batch.s),
            x=index_by_s(recon_list.x, batch.s),
            s=batch.s,
            recons_0=recon_list.x[0],
            recons_1=recon_list.x[1],
        )

    def configure_optimizers(
        self,
    ) -> tuple[list[torch.optim.Optimizer], list[CosineAnnealingWarmRestarts]]:
        ...

    def get_recon(self, dataloader: DataLoader) -> np.ndarray:
        raise NotImplementedError("This shouldn't be called. Only implementing for the abc.")

    def invert(self, z: Tensor, x: Tensor) -> Tensor:
        return z
