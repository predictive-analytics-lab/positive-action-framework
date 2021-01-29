"""Run the Autoencoder."""
import logging

import pandas as pd
from ethicml import (
    LRCV,
    TNR,
    TPR,
    Accuracy,
    Agarwal,
    DataTuple,
    Kamiran,
    Kamishima,
    Prediction,
    ProbPos,
    ZafarFairness,
    diff_per_sensitive_attribute,
    metric_per_sensitive_attribute,
    ratio_per_sensitive_attribute,
)
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger

from src.config_classes.dataclasses import Config
from src.data_modules.create import create_data_module
from src.ethicml_extension.oracle import DPOracle, EqOppOracle, Oracle
from src.logging import do_log
from src.model.aies_model import AiesModel
from src.model.classifier_model import Clf
from src.model.encoder_model import AE
from src.plotting import label_plot
from src.scoring import get_miri_metrics, produce_baselines
from src.utils import get_trainer, get_wandb_logger, produce_selection_groups

log = logging.getLogger(__name__)


def run_aies(cfg: Config) -> None:
    """Run the X Autoencoder."""
    seed_everything(0)  # cfg.data.seed)
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
    wandb_logger = get_wandb_logger(cfg)
    data.make_data_plots(data.cf_available, wandb_logger)

    # enc_early_stop_callback = EarlyStopping(monitor='val_mse', patience=30, mode="min")
    enc_trainer = get_trainer(
        cfg.training.gpus, wandb_logger, cfg.training.enc_epochs  # , callbacks=[enc_early_stop_callback]
    )
    enc_trainer.fit(encoder, datamodule=data)
    enc_trainer.test(ckpt_path=None, datamodule=data)

    # augmented_data = AugmentedDataModule(data, encoder)

    classifier = Clf(
        cfg.clf,
        num_s=data.num_s,
        data_dim=data.data_dim,
        s_dim=data.s_dim,
        cf_available=data.cf_available,
        outcome_cols=data.outcome_columns,
    )
    # clf_early_stop_callback = EarlyStopping(monitor='val_bce', patience=30, mode="min")
    clf_trainer = get_trainer(
        cfg.training.gpus, wandb_logger, cfg.training.clf_epochs  # , callbacks=[clf_early_stop_callback]
    )
    clf_trainer.fit(classifier, datamodule=data)
    clf_trainer.test(ckpt_path=None, datamodule=data)

    model = AiesModel(encoder=encoder, classifier=classifier)
    model_trainer = get_trainer(cfg.training.gpus, wandb_logger, 0)
    model_trainer.fit(model, datamodule=data)
    model_trainer.test(ckpt_path=None, datamodule=data)

    preds = produce_selection_groups(model.pd_results, data, model.recon_0, model.recon_1, wandb_logger)
    multiple_metrics(preds, data.test_data, "Ours-Post-Selection", wandb_logger)
    fair_preds = produce_selection_groups(model.pd_results, data, model.recon_0, model.recon_1, wandb_logger, fair=True)
    multiple_metrics(fair_preds, data.test_data, "Ours-Fair", wandb_logger)

    # === This is only for reporting ====
    data.flip_train_test()
    _model = AiesModel(encoder=encoder, classifier=classifier)
    _model_trainer = get_trainer(cfg.training.gpus, wandb_logger, 0)
    _model_trainer.fit(_model, datamodule=data)
    _model_trainer.test(ckpt_path=None, datamodule=data)
    produce_selection_groups(_model.pd_results, data, _model.recon_0, _model.recon_1, wandb_logger, "Train")
    data.flip_train_test()
    # === === ===

    our_clf_preds = Prediction(hard=pd.Series(model.all_preds.squeeze(-1).detach().cpu().numpy()))
    multiple_metrics(our_clf_preds, data.test_data, "Ours-Real-World-Preds", wandb_logger)
    produce_baselines(encoder=encoder, dm=data, logger=wandb_logger)
    produce_baselines(encoder=classifier, dm=data, logger=wandb_logger)

    if cfg.training.all_baselines:
        for model in [LRCV(), Oracle(), DPOracle(), EqOppOracle(), Kamiran(), ZafarFairness(), Kamishima(), Agarwal()]:
            log.info(f"=== {model.name} ===")
            try:
                results = model.run(data.train_data, data.test_data)
            except ValueError:
                continue
            multiple_metrics(results, data.test_data, model.name, wandb_logger)
            if data.cf_available:
                log.info(f"=== {model.name} and \"True\" Data ===")
                results = model.run(data.train_data, data.test_data)
                multiple_metrics(results, data.true_test_data, f"{model.name}-TrueLabels", wandb_logger)
                get_miri_metrics(
                    method=f"Miri/{model.name}",
                    acceptance=DataTuple(
                        x=data.test_data.x.copy(), s=data.test_data.s.copy(), y=results.hard.to_frame()
                    ),
                    graduated=data.true_test_data,
                    logger=wandb_logger,
                )
        multiple_metrics(
            preds,
            data.test_data,
            "Ours-Post-Selection",
            wandb_logger,
        )

        multiple_metrics(
            fair_preds,
            data.test_data,
            "Ours-Fair",
            wandb_logger,
        )

        multiple_metrics(
            our_clf_preds,
            data.test_data,
            "Ours-Real-World-Preds",
            wandb_logger,
        )
        if data.cf_available:
            get_miri_metrics(
                method="Miri/Ours-Post-Selection",
                acceptance=DataTuple(x=data.test_data.x.copy(), s=data.test_data.s.copy(), y=preds.hard.to_frame()),
                graduated=data.true_test_data,
                logger=wandb_logger,
            )
            get_miri_metrics(
                method="Miri/Ours-Real-World-Preds",
                acceptance=DataTuple(
                    x=data.test_data.x.copy(), s=data.test_data.s.copy(), y=our_clf_preds.hard.to_frame()
                ),
                graduated=data.true_test_data,
                logger=wandb_logger,
            )

    if cfg.training.log:
        wandb_logger.experiment.finish()


def multiple_metrics(preds: Prediction, target: DataTuple, name: str, logger: WandbLogger) -> None:
    """Get multiple metrics."""
    try:
        label_plot(target.replace(y=preds.hard.to_frame()), logger, name)
    except (IndexError, KeyError):
        pass

    for metric in [Accuracy(), ProbPos(), TPR(), TNR()]:
        general_str = f"Results/{name}/{metric.name}"
        do_log(general_str, metric.score(preds, target), logger)
        per_group = metric_per_sensitive_attribute(preds, target, metric)
        for key, result in per_group.items():
            do_log(f"{general_str}-{key}", result, logger)
        for key, result in diff_per_sensitive_attribute(per_group).items():
            do_log(f"{general_str}-Abs-Diff-{key}", result, logger)
        for key, result in ratio_per_sensitive_attribute(per_group).items():
            do_log("{g_str}-Ratio-{k}".format(g_str=general_str, k=key.replace('/', '\\')), result, logger)
