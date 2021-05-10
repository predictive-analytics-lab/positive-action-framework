"""Main script."""
import copy
import logging
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Final, Optional

import hydra
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
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import MISSING, DictConfig, OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger

from paf.base_templates.base_module import BaseDataModule
from paf.config_classes.paf.data_modules.configs import (
    LilliputDataModuleConf,
    SemiAdultDataModuleConf,
    SimpleAdultDataModuleConf,
    SimpleXDataModuleConf,
    ThirdWayDataModuleConf,
)
from paf.config_classes.paf.model.configs import AEConf, ClfConf
from paf.config_classes.pytorch_lightning.trainer.configs import TrainerConf
from paf.ethicml_extension.oracle import DPOracle
from paf.log_progress import do_log
from paf.model import AE
from paf.model.aies_model import AiesModel
from paf.plotting import label_plot
from paf.scoring import get_miri_metrics, produce_baselines
from paf.selection import produce_selection_groups

log = logging.getLogger(__name__)


@dataclass
class ExpConfig:
    """Experiment config."""

    lr: float = 1.0e-3
    weight_decay: float = 1.0e-3
    momentum: float = 0.9
    seed: int = 42
    log_offline: Optional[bool] = False
    tags: str = ""
    baseline: bool = False


@dataclass
class Config:
    """Base Config Schema."""

    _target_: str = "paf.main.Config"
    data: Any = MISSING
    enc: Any = MISSING  # put config files for this into `conf/model/`
    clf: Any = MISSING  # put config files for this into `conf/model/`
    trainer: Any = MISSING
    exp: ExpConfig = MISSING
    exp_group: Optional[str] = None


warnings.simplefilter(action='ignore', category=RuntimeWarning)

cs = ConfigStore.instance()
cs.store(name="config_schema", node=Config)  # General Schema
cs.store(name="trainer_schema", node=TrainerConf, package="trainer")
cs.store(name="enc_schema", node=AEConf, package="enc")
cs.store(name="clf_schema", node=ClfConf, package="clf")

data_package: Final[str] = "data"  # package:dir_within_config_path
data_group: Final[str] = "schema/data"  # group
cs.store(name="adult", node=SimpleAdultDataModuleConf, package=data_package, group=data_group)
cs.store(name="semi-synth", node=SemiAdultDataModuleConf, package=data_package, group=data_group)
cs.store(name="lilliput", node=LilliputDataModuleConf, package=data_package, group=data_group)
cs.store(name="synth", node=SimpleXDataModuleConf, package=data_package, group=data_group)
cs.store(name="third", node=ThirdWayDataModuleConf, package=data_package, group=data_group)


@hydra.main(config_path="configs", config_name="base_conf")
def launcher(hydra_config: DictConfig) -> None:
    """Instantiate with hydra and get the experiments running!"""
    # hydra_config.data.data_dir = Path(hydra_config.data.data_dir).expanduser()
    cfg: Config = instantiate(hydra_config, _recursive_=True, _convert_="partial")
    run_aies(cfg, raw_config=OmegaConf.to_container(hydra_config, resolve=True, enum_to_str=True))


def run_aies(cfg: Config, raw_config: Optional[Dict[str, Any]]) -> None:
    """Run the X Autoencoder."""
    seed_everything(0)
    data: BaseDataModule = cfg.data
    data.prepare_data()

    log.info(f"data_dim={data.data_dim}, num_s={data.num_s}")
    encoder: AE = cfg.enc
    encoder.build(
        num_s=data.num_s,
        data_dim=data.data_dim,
        s_dim=data.s_dim,
        cf_available=data.cf_available,
        feature_groups=data.feature_groups,
        outcome_cols=data.column_names,
        scaler=data.scaler,
    )
    wandb_logger = (
        None
        if cfg.exp.log_offline
        else WandbLogger(
            entity="predictive-analytics-lab",
            project="paf",
            tags=cfg.exp.tags.split("/")[:-1],
            config=raw_config,
        )
    )
    cfg.trainer.logger = wandb_logger

    clf_trainer = copy.deepcopy(cfg.trainer)
    model_trainer = copy.deepcopy(cfg.trainer)
    _model_trainer = copy.deepcopy(cfg.trainer)

    data.make_data_plots(data.cf_available, cfg.trainer.logger)

    enc_trainer = cfg.trainer
    enc_trainer.tune(model=encoder, datamodule=data)
    enc_trainer.fit(model=encoder, datamodule=data)
    enc_trainer.test(model=encoder, ckpt_path=None, datamodule=data)

    classifier = cfg.clf
    classifier.build(
        num_s=data.num_s,
        data_dim=data.data_dim,
        s_dim=data.s_dim,
        cf_available=data.cf_available,
        feature_groups=data.feature_groups,
        outcome_cols=data.outcome_columns,
    )
    clf_trainer.tune(model=classifier, datamodule=data)
    clf_trainer.fit(model=classifier, datamodule=data)
    clf_trainer.test(model=classifier, ckpt_path=None, datamodule=data)

    model = AiesModel(encoder=encoder, classifier=classifier)
    model_trainer.test(model=model, ckpt_path=None, datamodule=data)

    preds = produce_selection_groups(
        model.pd_results, data, model.recon_0, model.recon_1, wandb_logger
    )
    multiple_metrics(
        preds,
        DataTuple(
            x=pd.DataFrame(model.all_x.cpu().numpy(), columns=data.test_data.x.columns),
            s=pd.DataFrame(model.all_s.cpu().numpy(), columns=data.test_data.s.columns),
            y=pd.DataFrame(model.all_y.cpu().numpy(), columns=data.test_data.y.columns),
        ),
        "Ours-Post-Selection",
        wandb_logger,
    )
    fair_preds = produce_selection_groups(
        model.pd_results, data, model.recon_0, model.recon_1, wandb_logger, fair=True
    )
    multiple_metrics(
        fair_preds,
        DataTuple(
            x=pd.DataFrame(model.all_x.cpu().numpy(), columns=data.test_data.x.columns),
            s=pd.DataFrame(model.all_s.cpu().numpy(), columns=data.test_data.s.columns),
            y=pd.DataFrame(model.all_y.cpu().numpy(), columns=data.test_data.y.columns),
        ),
        "Ours-Fair",
        wandb_logger,
    )

    # === This is only for reporting ====
    data.flip_train_test()
    _model = AiesModel(encoder=encoder, classifier=classifier)
    _model_trainer.test(model=_model, ckpt_path=None, datamodule=data)
    produce_selection_groups(
        _model.pd_results, data, _model.recon_0, _model.recon_1, wandb_logger, "Train"
    )
    data.flip_train_test()
    # === === ===

    our_clf_preds = Prediction(hard=pd.Series(model.all_preds.squeeze(-1).detach().cpu().numpy()))
    multiple_metrics(
        our_clf_preds,
        DataTuple(
            x=pd.DataFrame(model.all_x.cpu().numpy(), columns=data.test_data.x.columns),
            s=pd.DataFrame(model.all_s.cpu().numpy(), columns=data.test_data.s.columns),
            y=pd.DataFrame(model.all_y.cpu().numpy(), columns=data.test_data.y.columns),
        ),
        "Ours-Real-World-Preds",
        wandb_logger,
    )
    produce_baselines(encoder=encoder, dm=data, logger=wandb_logger)
    produce_baselines(encoder=classifier, dm=data, logger=wandb_logger)

    if cfg.exp.baseline:
        for model in [
            LRCV(),
            # Oracle(),
            DPOracle(),
            # EqOppOracle(),
            # Kamiran(),
            # ZafarFairness(),
            # Kamishima(),
            # Agarwal(),
        ]:
            log.info(f"=== {model.name} ===")
            try:
                results = model.run(data.train_data, data.test_data)
            except ValueError:
                continue
            multiple_metrics(results, data.test_data, model.name, wandb_logger)
            if data.cf_available:
                log.info(f"=== {model.name} and \"True\" Data ===")
                results = model.run(data.train_data, data.test_data)
                multiple_metrics(
                    results, data.true_test_data, f"{model.name}-TrueLabels", wandb_logger
                )
                get_miri_metrics(
                    method=f"Miri/{model.name}",
                    acceptance=DataTuple(
                        x=data.test_data.x.copy(),
                        s=data.test_data.s.copy(),
                        y=results.hard.to_frame(),
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
                acceptance=DataTuple(
                    x=data.test_data.x.copy(), s=data.test_data.s.copy(), y=preds.hard.to_frame()
                ),
                graduated=data.true_test_data,
                logger=wandb_logger,
            )
            get_miri_metrics(
                method="Miri/Ours-Fair",
                acceptance=DataTuple(
                    x=data.test_data.x.copy(),
                    s=data.test_data.s.copy(),
                    y=fair_preds.hard.to_frame(),
                ),
                graduated=data.true_test_data,
                logger=wandb_logger,
            )
            get_miri_metrics(
                method="Miri/Ours-Real-World-Preds",
                acceptance=DataTuple(
                    x=data.test_data.x.copy(),
                    s=data.test_data.s.copy(),
                    y=our_clf_preds.hard.to_frame(),
                ),
                graduated=data.true_test_data,
                logger=wandb_logger,
            )

    # if not cfg.exp.log_offline:
    #     wandb_logger.experiment.finish()


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
            do_log(
                "{g_str}-Ratio-{k}".format(g_str=general_str, k=key.replace('/', '\\')),
                result,
                logger,
            )


if __name__ == '__main__':
    launcher()
