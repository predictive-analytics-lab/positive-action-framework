"""Run the Autoencoder."""
import logging

import pandas as pd
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
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger

from src.config_classes.dataclasses import Config
from src.data_modules.create import create_data_module
from src.model.aies_model import AiesModel
from src.model.classifier_model import Clf
from src.model.encoder_model import AE
from src.scoring import produce_baselines
from src.utils import do_log, flatten, produce_selection_groups

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
    if cfg.training.gpus > 0:
        enc_trainer = Trainer(
            gpus=cfg.training.gpus,
            max_epochs=cfg.training.enc_epochs,
            logger=wandb_logger,
            deterministic=True,
        )
    else:
        enc_trainer = Trainer(
            max_epochs=cfg.training.enc_epochs,
            logger=wandb_logger,
            deterministic=True,
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
    if cfg.training.gpus > 0:
        clf_trainer = Trainer(
            gpus=cfg.training.gpus,
            max_epochs=cfg.training.clf_epochs,
            logger=wandb_logger,
            deterministic=True,
        )
    else:
        clf_trainer = Trainer(
            max_epochs=cfg.training.clf_epochs,
            logger=wandb_logger,
            deterministic=True,
        )
    clf_trainer.fit(classifier, datamodule=data)
    clf_trainer.test(ckpt_path=None, datamodule=data)

    model = AiesModel(encoder=encoder, classifier=classifier)
    if cfg.training.gpus > 0:
        model_trainer = Trainer(
            gpus=cfg.training.gpus,
            max_epochs=0,
            deterministic=True,
            logger=wandb_logger,
        )
    else:
        model_trainer = Trainer(max_epochs=0, deterministic=True, logger=wandb_logger)
    model_trainer.fit(model, datamodule=data)
    model_trainer.test(ckpt_path=None, datamodule=data)

    preds = produce_selection_groups(model.pd_results, wandb_logger)

    multiple_metrics(preds, data.test_data, "Ours-Post-Selection", wandb_logger)
    multiple_metrics(
        Prediction(hard=pd.Series(model.all_preds.squeeze(-1).detach().cpu().numpy())),
        data.test_data,
        "Ours-Real-World-Preds",
        wandb_logger,
    )

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
        general_str = f"Results/{name}-{metric.name}"
        do_log(general_str, metric.score(preds, target), logger)
        per_group = metric_per_sensitive_attribute(preds, target, metric)
        for key, result in per_group.items():
            do_log(f"{general_str}-{key}", result, logger)
        for key, result in diff_per_sensitive_attribute(per_group).items():
            do_log(f"{general_str}-Abs-Diff-{key}", result, logger)
        for key, result in ratio_per_sensitive_attribute(per_group).items():
            do_log(f"{general_str}-Ratio-{key}", result, logger)
