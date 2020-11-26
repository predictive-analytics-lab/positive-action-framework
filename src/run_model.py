"""Run the Autoencoder."""
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from ethicml import (
    LRCV,
    TNR,
    TPR,
    Accuracy,
    DataTuple,
    Prediction,
    ProbPos,
    diff_per_sensitive_attribute,
    metric_per_sensitive_attribute,
    ratio_per_sensitive_attribute,
)
from omegaconf import OmegaConf
from pytorch_lightning import LightningDataModule, Trainer, seed_everything
from pytorch_lightning.loggers import LightningLoggerBase, WandbLogger
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import KFold

import wandb
from src.config_classes.dataclasses import Config
from src.data_modules.base_module import BaseDataModule
from src.data_modules.create import create_data_module
from src.model.aies_model import AiesModel
from src.model.classifier_model import Clf
from src.model.common_model import CommonModel
from src.model.encoder_model import AE
from src.utils import do_log, facct_mapper, flatten, selection_rules

log = logging.getLogger(__name__)


def run_aies(cfg: Config) -> None:
    """Run the X Autoencoder."""
    seed_everything(cfg.data.seed)
    data = create_data_module(cfg.data)
    data.prepare_data()
    log.info(f"data_dim={data.data_dim}, num_s={data.num_s}")
    encoder = AE(
        cfg.enc,
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
        config=flatten(OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)),
    )
    enc_trainer = Trainer(
        max_epochs=cfg.training.enc_epochs, logger=wandb_logger, deterministic=True
    )
    enc_trainer.fit(encoder, datamodule=data)
    enc_trainer.test(ckpt_path=None, datamodule=data)

    classifier = Clf(
        cfg.clf,
        num_s=data.num_s,
        data_dim=data.data_dim,
        s_dim=data.s_dim,
        cf_available=data.cf_available,
        outcome_cols=data.outcome_columns,
    )
    clf_trainer = Trainer(
        max_epochs=cfg.training.clf_epochs, logger=wandb_logger, deterministic=True
    )
    clf_trainer.fit(classifier, datamodule=data)
    clf_trainer.test(ckpt_path=None, datamodule=data)

    model = AiesModel(encoder=encoder, classifier=classifier)
    model_trainer = Trainer(max_epochs=0, deterministic=True)
    model_trainer.fit(model, datamodule=data)
    model_trainer.test(ckpt_path=None, datamodule=data)

    preds = produce_selection_groups(model.pd_results, wandb_logger)
    score_preds(preds, data, wandb_logger, "Ours-Post-Selection")
    score_preds(
        Prediction(hard=pd.Series(model.all_preds.squeeze(-1).detach().cpu().numpy())),
        data,
        wandb_logger,
        "Ours-Real-World-Preds",
    )

    multiple_metrics(preds, data.test_data, "Ours-Post-Selection", wandb_logger)

    produce_baselines(encoder=encoder, dm=data, logger=wandb_logger)
    produce_baselines(encoder=classifier, dm=data, logger=wandb_logger)

    if cfg.training.all_baselines:
        for model in [LRCV()]:
            log.info(f"=== {model.name} ===")
            results = model.run(data.train_data, data.test_data)
            multiple_metrics(results, data.test_data, model.name, wandb_logger)

    wandb_logger.experiment.finish()


def multiple_metrics(preds: Prediction, target: DataTuple, name: str, logger: WandbLogger) -> None:
    """Get multiple metrics."""
    for metric in [Accuracy(), ProbPos(), TPR(), TNR()]:
        general_str = f"Results-{name}-{metric.name}"
        do_log(general_str, metric.score(preds, target), logger)
        per_group = metric_per_sensitive_attribute(preds, target, metric)
        for key, result in per_group.items():
            do_log(f"{general_str}-{key}", result, logger)
        for key, result in diff_per_sensitive_attribute(per_group).items():
            do_log(f"{general_str}-Abs-Diff-{key}", result, logger)
        for key, result in ratio_per_sensitive_attribute(per_group).items():
            do_log(f"{general_str}-Ratio-{key}", result, logger)

    for metric in [Accuracy(), ProbPos(), TPR(), TNR()]:
        do_log(general_str, metric.score(preds, target), logger)
        for k, v in metric_per_sensitive_attribute(preds, target, metric).items():
            log.info(f"{metric.name}-{k}: {v}")


def score_preds(
    preds: Prediction, dm: LightningDataModule, logger: WandbLogger, clf_name: str
) -> None:
    """Score the predictions."""
    res_name = f"Baselines/{clf_name}-Accuracy-Y-from-X"
    score = Accuracy().score(prediction=preds, actual=dm.test_data)
    do_log(res_name, score, logger)


def produce_selection_groups(outcomes: pd.DataFrame, logger: LightningLoggerBase) -> Prediction:
    """Follow Selection rules."""
    outcomes_hist(outcomes, logger)
    outcomes["decision"] = selection_rules(outcomes)
    for idx, val in outcomes["decision"].value_counts().iteritems():
        do_log(f"Table3/selection_rule_group_{idx}", val, logger)
    return facct_mapper(Prediction(hard=outcomes["decision"]))


def outcomes_hist(outcomes: pd.DataFrame, logger: WandbLogger) -> None:
    """Produce a distribution of the outcomes."""
    val_counts = (
        outcomes[["s1_0_s2_0", "s1_0_s2_1", "s1_1_s2_0", "s1_1_s2_1"]].sum(axis=1).value_counts()
    )
    sns.barplot(val_counts.index, val_counts.values)
    logger.experiment.log({"Debugging2/Outcomes": wandb.Plotly(plt)})
    for idx, val in val_counts.iteritems():
        do_log(f"Debugging2/Outcomes-{idx}", val, logger)
    plt.clf()


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


def produce_baselines(
    *, encoder: CommonModel, dm: BaseDataModule, logger: LightningLoggerBase
) -> None:
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
