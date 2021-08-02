"""Main script."""
import copy
from dataclasses import dataclass
from enum import Enum, auto
import logging
from typing import Any, Final, Optional
import warnings

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
import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import DictConfig, MISSING, OmegaConf
import pandas as pd
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger

from paf.base_templates.base_module import BaseDataModule
from paf.config_classes.bolts.fair.data.configs import (
    AdmissionsDataModuleConf,
    AdultDataModuleConf,
    CrimeDataModuleConf,
    HealthDataModuleConf,
    LawDataModuleConf,
)
from paf.config_classes.paf.data_modules.configs import (  # type: ignore[import]
    LilliputDataModuleConf,
    SemiAdultDataModuleConf,
    SimpleAdultDataModuleConf,
    SimpleXDataModuleConf,
    ThirdWayDataModuleConf,
)
from paf.config_classes.paf.model.configs import (  # type: ignore[import]
    AEConf,
    ClfConf,
    CycleGanConf,
)
from paf.config_classes.pytorch_lightning.trainer.configs import (
    TrainerConf,  # type: ignore[import]
)
from paf.ethicml_extension.oracle import DPOracle
from paf.log_progress import do_log
from paf.model import AE
from paf.model.aies_model import AiesModel
from paf.model.naive import NaiveModel
from paf.model.nearestneighbour_model import NearestNeighbourModel
from paf.plotting import label_plot, make_data_plots
from paf.scoring import get_miri_metrics, produce_baselines
from paf.selection import baseline_selection_rules, produce_selection_groups

log = logging.getLogger(__name__)


class ModelType(Enum):
    paf = auto()
    nn = auto()


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
    model: ModelType = ModelType.paf


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
cs.store(name="clf_schema", node=ClfConf, package="clf")

enc_package: Final[str] = "enc"
enc_group: Final[str] = "schema/enc"
cs.store(name="enc_schema", node=AEConf, package=enc_package, group=enc_group)
cs.store(name="cyclegan", node=CycleGanConf, package=enc_package, group=enc_group)

data_package: Final[str] = "data"  # package:dir_within_config_path
data_group: Final[str] = "schema/data"  # group
cs.store(name="adult-bolt", node=AdultDataModuleConf, package=data_package, group=data_group)
cs.store(name="admiss-bolt", node=AdmissionsDataModuleConf, package=data_package, group=data_group)
cs.store(name="crime-bolt", node=CrimeDataModuleConf, package=data_package, group=data_group)
cs.store(name="health-bolt", node=HealthDataModuleConf, package=data_package, group=data_group)
cs.store(name="law-bolt", node=LawDataModuleConf, package=data_package, group=data_group)
cs.store(name="adult", node=SimpleAdultDataModuleConf, package=data_package, group=data_group)
cs.store(name="semi-synth", node=SemiAdultDataModuleConf, package=data_package, group=data_group)
cs.store(name="lilliput", node=LilliputDataModuleConf, package=data_package, group=data_group)
cs.store(name="synth", node=SimpleXDataModuleConf, package=data_package, group=data_group)
cs.store(name="third", node=ThirdWayDataModuleConf, package=data_package, group=data_group)


@hydra.main(config_path="configs", config_name="base_conf")
def launcher(hydra_config: DictConfig) -> None:
    """Instantiate with hydra and get the experiments running!"""
    cfg: Config = instantiate(hydra_config, _recursive_=True, _convert_="partial")
    run_aies(cfg, raw_config=OmegaConf.to_container(hydra_config, resolve=True, enum_to_str=True))


def run_aies(cfg: Config, raw_config: Any) -> None:
    """Run the X Autoencoder."""
    seed_everything(cfg.exp.seed)
    data: BaseDataModule = cfg.data
    data.prepare_data()
    data.setup()

    log.info(f"data_dim={data.size()}, num_s={data.card_s}")

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

    make_data_plots(data, cfg.trainer.logger)

    # data.make_data_plots(data.cf_available, cfg.trainer.logger)

    if cfg.exp.model == ModelType.paf:

        encoder: AE = cfg.enc
        encoder.build(
            num_s=data.card_s,
            data_dim=data.size()[0],
            s_dim=data.dim_s[0],
            cf_available=data.cf_available if hasattr(data, "cf_available") else False,
            feature_groups=data.feature_groups,
            outcome_cols=data.disc_features + data.cont_features,
            scaler=data.scaler,
        )

        enc_trainer = cfg.trainer
        # enc_trainer.tune(model=encoder, datamodule=data)
        enc_trainer.fit(model=encoder, datamodule=data)
        if enc_trainer.fast_dev_run:
            enc_trainer.test(model=encoder, datamodule=data, ckpt_path=None)
        else:
            enc_trainer.test(datamodule=data)

        classifier = cfg.clf
        classifier.build(
            num_s=data.card_s,
            data_dim=data.size()[0],
            s_dim=data.dim_s[0],
            cf_available=data.cf_available if hasattr(data, "cf_available") else False,
            feature_groups=data.feature_groups,
            outcome_cols=data.disc_features + data.cont_features,
        )
        clf_trainer.tune(model=classifier, datamodule=data)
        clf_trainer.fit(model=classifier, datamodule=data)
        if clf_trainer.fast_dev_run:
            clf_trainer.test(model=classifier, ckpt_path=None, datamodule=data)
        else:
            clf_trainer.test(datamodule=data)

        model = AiesModel(encoder=encoder, classifier=classifier)

    elif cfg.exp.model == ModelType.nn:
        classifier = NaiveModel(in_size=cfg.data.dim_x[0])
        clf_trainer.tune(model=classifier, datamodule=data)
        clf_trainer.fit(model=classifier, datamodule=data)

        model = NearestNeighbourModel(clf_model=classifier, data=data)

    model_trainer.fit(model=model, datamodule=data)
    if model_trainer.fast_dev_run:
        model_trainer.test(model=model, ckpt_path=None, datamodule=data)
    else:
        model_trainer.test(datamodule=data)

    if cfg.exp.model == ModelType.paf:
        preds = produce_selection_groups(
            model.pd_results, data, model.recon_0, model.recon_1, wandb_logger
        )
        multiple_metrics(
            preds,
            DataTuple(
                x=pd.DataFrame(model.all_x.cpu().numpy(), columns=data.test_datatuple.x.columns),
                s=pd.DataFrame(model.all_s.cpu().numpy(), columns=data.test_datatuple.s.columns),
                y=pd.DataFrame(model.all_y.cpu().numpy(), columns=data.test_datatuple.y.columns),
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
                x=pd.DataFrame(model.all_x.cpu().numpy(), columns=data.test_datatuple.x.columns),
                s=pd.DataFrame(model.all_s.cpu().numpy(), columns=data.test_datatuple.s.columns),
                y=pd.DataFrame(model.all_y.cpu().numpy(), columns=data.test_datatuple.y.columns),
            ),
            "Ours-Fair",
            wandb_logger,
        )

        # === This is only for reporting ====
        _model = AiesModel(encoder=encoder, classifier=classifier)
        _model_trainer.test(model=_model, ckpt_path=None, dataloaders=data.train_dataloader())
        produce_selection_groups(
            _model.pd_results, data, _model.recon_0, _model.recon_1, wandb_logger, "Train"
        )
        # === === ===

        our_clf_preds = Prediction(
            hard=pd.Series(model.all_preds.squeeze(-1).detach().cpu().numpy())
        )
        multiple_metrics(
            our_clf_preds,
            DataTuple(
                x=pd.DataFrame(model.all_x.cpu().numpy(), columns=data.test_datatuple.x.columns),
                s=pd.DataFrame(model.all_s.cpu().numpy(), columns=data.test_datatuple.s.columns),
                y=pd.DataFrame(model.all_y.cpu().numpy(), columns=data.test_datatuple.y.columns),
            ),
            "Ours-Real-World-Preds",
            wandb_logger,
        )
        if isinstance(cfg.enc, AE):
            produce_baselines(
                encoder=encoder, dm=data, logger=wandb_logger, test_mode=cfg.trainer.fast_dev_run
            )
            produce_baselines(
                encoder=classifier, dm=data, logger=wandb_logger, test_mode=cfg.trainer.fast_dev_run
            )

    else:
        preds = baseline_selection_rules(model.pd_results, wandb_logger)

    if cfg.exp.baseline:
        for model in [
            NaiveModel(in_size=data.data_dim, num_pos_action=2),
            # LRCV(),
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
                results = model.run(data.train_datatuple, data.test_datatuple)
            except ValueError:
                continue
            multiple_metrics(results, data.test_datatuple, model.name, wandb_logger)
            if data.cf_available:
                log.info(f"=== {model.name} and \"True\" Data ===")
                results = model.run(data.train_datatuple, data.test_datatuple)
                multiple_metrics(
                    results, data.true_test_datatuple, f"{model.name}-TrueLabels", wandb_logger
                )
                get_miri_metrics(
                    method=f"Miri/{model.name}",
                    acceptance=DataTuple(
                        x=data.test_datatuple.x.copy(),
                        s=data.test_datatuple.s.copy(),
                        y=results.hard.to_frame(),
                    ),
                    graduated=data.true_test_datatuple,
                    logger=wandb_logger,
                )
        multiple_metrics(
            preds,
            data.test_datatuple,
            "Ours-Post-Selection",
            wandb_logger,
        )

        multiple_metrics(
            fair_preds,
            data.test_datatuple,
            "Ours-Fair",
            wandb_logger,
        )

        multiple_metrics(
            our_clf_preds,
            data.test_datatuple,
            "Ours-Real-World-Preds",
            wandb_logger,
        )
        if data.cf_available:
            get_miri_metrics(
                method="Miri/Ours-Post-Selection",
                acceptance=DataTuple(
                    x=data.test_datatuple.x.copy(),
                    s=data.test_datatuple.s.copy(),
                    y=preds.hard.to_frame(),
                ),
                graduated=data.true_test_datatuple,
                logger=wandb_logger,
            )
            get_miri_metrics(
                method="Miri/Ours-Fair",
                acceptance=DataTuple(
                    x=data.test_datatuple.x.copy(),
                    s=data.test_datatuple.s.copy(),
                    y=fair_preds.hard.to_frame(),
                ),
                graduated=data.true_test_datatuple,
                logger=wandb_logger,
            )
            get_miri_metrics(
                method="Miri/Ours-Real-World-Preds",
                acceptance=DataTuple(
                    x=data.test_datatuple.x.copy(),
                    s=data.test_datatuple.s.copy(),
                    y=our_clf_preds.hard.to_frame(),
                ),
                graduated=data.true_test_datatuple,
                logger=wandb_logger,
            )

    if not cfg.exp.log_offline:
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
            do_log(
                "{g_str}-Ratio-{k}".format(g_str=general_str, k=key.replace('/', '\\')),
                result,
                logger,
            )


if __name__ == '__main__':
    launcher()
