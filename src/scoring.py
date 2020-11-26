"""Scoring functions."""
import numpy as np
import pandas as pd
from ethicml import Accuracy, Prediction
from pytorch_lightning.loggers import LightningLoggerBase
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import KFold

from src.data_modules.base_module import BaseDataModule
from src.model.common_model import CommonModel
from src.model.encoder_model import AE
from src.utils import do_log


def lrcv_results(
    train: np.ndarray,
    test: np.ndarray,
    dm: BaseDataModule,
    logger: LightningLoggerBase,
    component: str,
) -> None:
    """Run an LRCV over some train set and apply to some test set."""
    for train_target, test_target, target_name in [
        (dm.train_data.s, dm.test_data.s, "S"),
        (dm.train_data.y, dm.test_data.y, "Y"),
    ]:
        random_state = np.random.RandomState(888)
        folder = KFold(n_splits=5, shuffle=True, random_state=random_state)
        clf = LogisticRegressionCV(
            cv=folder,
            n_jobs=-1,
            random_state=random_state,
            solver="liblinear",
            multi_class="auto",
        )
        clf.fit(train, train_target.to_numpy().ravel())
        s_preds = Prediction(hard=pd.Series(clf.predict(test)), info=dict(C=clf.C_[0]))
        do_log(
            f"Baselines/Accuracy-{target_name}-from-{component}",
            Accuracy().score(prediction=s_preds, actual=dm.test_data.replace(y=test_target)),
            logger,
        )


def produce_baselines(*, encoder: CommonModel, dm: BaseDataModule, logger: LightningLoggerBase) -> None:
    """Produce baselines for predictiveness."""
    latent_train = encoder.get_latent(dm.train_dataloader(shuffle=False, drop_last=False))
    latent_test = encoder.get_latent(dm.test_dataloader())
    lrcv_results(latent_train, latent_test, dm, logger, "Enc-Z")

    if isinstance(encoder, AE):
        train = dm.train_data.x.to_numpy()
        test = dm.test_data.x.to_numpy()
        lrcv_results(train, test, dm, logger, "Og-Data")
        recon_name = "Recon-Data"
    else:
        train_labels = dm.train_data.y.to_numpy()
        test_labels = dm.test_data.y.to_numpy()
        lrcv_results(train_labels, test_labels, dm, logger, "Og-Labels")
        recon_name = "Preds"

    train_recon = encoder.get_recon(dm.train_dataloader(shuffle=False, drop_last=False))
    test_recon = encoder.get_recon(dm.test_dataloader())
    lrcv_results(train_recon, test_recon, dm, logger, recon_name)
