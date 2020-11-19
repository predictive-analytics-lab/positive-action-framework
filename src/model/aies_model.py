"""AIES Model."""
from ethicml import implements
from pytorch_lightning import LightningDataModule, LightningModule
from torch import Tensor

from src.model.classifier_model import Clf
from src.model.encoder_model import AE


class AiesModel(LightningModule):
    """Model."""

    def __init__(self, encoder: AE, classifier: Clf):
        super().__init__()
        self.enc = encoder
        self.clf = classifier

    @implements(LightningModule)
    def forward(self, x: Tensor, s: Tensor):
        enc_z, enc_s_pred, recons = self.enc(x, s)
        preds_dict = self.clf.from_recons(recons)
        return preds_dict

    def do_run(self, dm: LightningDataModule):
        """Run the enc and clf end-to-end."""
        preds = {}
        for x, s, y, _, _, _ in dm.test_dataloader():
            asdf = self(x, s)
            for k, v in asdf.items:
                z, s_pred, y_pred = v
                if k not in preds.keys():
                    preds[k] = {"z": z, "s_pred": s_pred, "y_pred": y_pred}
                else:
                    preds[k] = preds[k] + v

        return preds
