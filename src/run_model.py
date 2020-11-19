"""Run the Autoencoder."""
import logging

import numpy as np
import pandas as pd
from ethicml import Accuracy, Prediction
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import LightningLoggerBase, WandbLogger
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import KFold

from src.config_classes.dataclasses import Config
from src.data_modules.base_module import BaseDataModule
from src.data_modules.create import create_data_module
from src.model.aies_model import AiesModel
from src.model.classifier_model import Clf
from src.model.encoder_model import AE

log = logging.getLogger(__name__)


def run_aies(cfg: Config) -> None:
    """Run the X Autoencoder."""
    seed_everything(cfg.data.seed)
    data = create_data_module(cfg.data)
    data.prepare_data()
    log.info(f"data_dim={data.data_dim}, num_s={data.num_s}")
    encoder = AE(
        cfg.model,
        num_s=data.num_s,
        data_dim=data.data_dim,
        s_dim=data.s_dim,
        cf_available=data.cf_available,
        feature_groups=data.feature_groups,
        column_names=data.column_names,
    )
    wandb_logger = WandbLogger(
        entity="predictive-analytics-lab",
        project="aies",
        tags=cfg.training.tags.split("/")[:-1],
    )
    enc_trainer = Trainer(max_epochs=cfg.training.epochs, logger=wandb_logger, deterministic=True)
    enc_trainer.fit(encoder, datamodule=data)
    enc_trainer.test(ckpt_path=None, datamodule=data)

    classifier = Clf(
        cfg.model,
        num_s=data.num_s,
        data_dim=data.data_dim,
        s_dim=data.s_dim,
        cf_available=data.cf_available,
        outcome_cols=data.outcome_columns,
    )
    clf_trainer = Trainer(max_epochs=cfg.training.epochs, logger=wandb_logger, deterministic=True)
    clf_trainer.fit(classifier, datamodule=data)
    clf_trainer.test(ckpt_path=None, datamodule=data)

    model = AiesModel(encoder=encoder, classifier=classifier)

    # model.do_run(data)
    produce_baselines(encoder=encoder, dm=data, logger=wandb_logger)


def lrcv_results(
    train: np.ndarray,
    test: np.ndarray,
    dm: BaseDataModule,
    logger: LightningLoggerBase,
    component: str,
) -> None:
    """Run an LRCV over some train set and apply to some test set."""
    random_state = np.random.RandomState(888)
    folder = KFold(n_splits=5, shuffle=True, random_state=random_state)
    clf = LogisticRegressionCV(
        cv=folder,
        n_jobs=-1,
        random_state=random_state,
        solver="liblinear",
        multi_class="auto",
    )
    clf.fit(train, dm.train_data.s.to_numpy().ravel())
    s_preds = Prediction(hard=pd.Series(clf.predict(test)), info=dict(C=clf.C_[0]))
    logger.experiment.log(
        {
            f"Baselines/Accuracy-S-from-{component}": Accuracy().score(
                prediction=s_preds, actual=dm.test_data.replace(y=dm.test_data.s)
            )
        }
    )
    log.info(
        f"Accuracy-S-from-{component}: "
        f"{Accuracy().score(prediction=s_preds, actual=dm.test_data.replace(y=dm.test_data.s))}",
    )


def produce_baselines(*, encoder: AE, dm: BaseDataModule, logger: LightningLoggerBase) -> None:
    """Produce baselines for predictiveness."""
    latent_train = encoder.get_latent(dm.train_dataloader(shuffle=False, drop_last=False))
    latent_test = encoder.get_latent(dm.test_dataloader())
    lrcv_results(latent_train, latent_test, dm, logger, "Enc-Z")

    train = dm.train_data.x.to_numpy()
    test = dm.test_data.x.to_numpy()
    lrcv_results(train, test, dm, logger, "Og-Data")

    train_recon = encoder.get_recon(dm.train_dataloader(shuffle=False, drop_last=False))
    test_recon = encoder.get_recon(dm.test_dataloader())
    lrcv_results(train_recon, test_recon, dm, logger, "Recon-Data")
