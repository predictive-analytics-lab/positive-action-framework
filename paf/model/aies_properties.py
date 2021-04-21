"""Properties for the AIES model."""

from typing import Optional

from pytorch_lightning import LightningModule
from torch import Tensor


class AiesProperties(LightningModule):
    """Properties for the  AIES model."""

    def __init__(self) -> None:
        super().__init__()
        self.all_enc_z: Optional[Tensor] = None
        self.all_enc_s_pred: Optional[Tensor] = None
        self._all_s: Optional[Tensor] = None
        self._all_x: Optional[Tensor] = None
        self._all_y: Optional[Tensor] = None
        self._all_recon: Optional[Tensor] = None
        self._all_preds: Optional[Tensor] = None

    @property
    def all_preds(self) -> Tensor:
        assert self._all_preds is not None
        return self._all_preds

    @all_preds.setter
    def all_preds(self, all_preds: Tensor) -> None:
        self._all_preds = all_preds

    @property
    def all_s(self) -> Tensor:
        assert self._all_s is not None
        return self._all_s

    @all_s.setter
    def all_s(self, all_s: Tensor) -> None:
        self._all_s = all_s

    @property
    def all_x(self) -> Tensor:
        assert self._all_x is not None
        return self._all_x

    @all_x.setter
    def all_x(self, all_x: Tensor) -> None:
        self._all_x = all_x

    @property
    def all_y(self) -> Tensor:
        assert self._all_y is not None
        return self._all_y

    @all_y.setter
    def all_y(self, all_y: Tensor) -> None:
        self._all_y = all_y

    @property
    def all_recon(self) -> Tensor:
        assert self._all_recon is not None
        return self._all_recon

    @all_recon.setter
    def all_recon(self, all_recon: Tensor) -> None:
        self._all_recon = all_recon

    @property
    def all_enc_z(self) -> Tensor:
        assert self._all_enc_z is not None
        return self._all_enc_z

    @all_enc_z.setter
    def all_enc_z(self, all_enc_z: Tensor) -> None:
        self._all_enc_z = all_enc_z

    @property
    def all_enc_s_pred(self) -> Tensor:
        assert self._all_enc_s_pred is not None
        return self._all_enc_s_pred

    @all_enc_s_pred.setter
    def all_enc_s_pred(self, all_enc_s_pred: Tensor) -> None:
        self._all_enc_s_pred = all_enc_s_pred
