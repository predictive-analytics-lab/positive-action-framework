"""Main script."""
from __future__ import annotations
from copy import copy

if 1:
    import faiss  # noqa

from dataclasses import dataclass
from enum import Enum, auto
import logging
from typing import Any, Final, List, Optional
import warnings

from conduit.hydra.conduit.fair.data.datamodules.conf import (  # type: ignore[import]
    AdmissionsDataModuleConf,
    AdultDataModuleConf,
    CompasDataModuleConf,
    CreditDataModuleConf,
    CrimeDataModuleConf,
    HealthDataModuleConf,
    LawDataModuleConf,
    SqfDataModuleConf,
)
import ethicml as em
import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, MISSING, OmegaConf
import pandas as pd
import pytorch_lightning as pl
import pytorch_lightning.loggers as pll
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import umap

from paf.architectures import PafModel, PafResults, Results
from paf.architectures.model.model_components import AE
from paf.base_templates.base_module import BaseDataModule
from paf.callbacks.callbacks import L1Logger
from paf.config_classes.ethicml.configs import (  # type: ignore[import]
    AgarwalConf,
    DPOracleConf,
    KamiranConf,
    KamishimaConf,
    LRCVConf,
    OracleConf,
    ZafarFairnessConf,
)
from paf.config_classes.paf.architectures.model.configs import (  # type: ignore[import]
    CycleGanConf,
    NearestNeighbourConf,
)
from paf.config_classes.paf.architectures.model.model_components.configs import (  # type: ignore[import]
    AEConf,
    ClfConf,
)
from paf.config_classes.paf.data_modules.configs import (  # type: ignore[import]
    LilliputDataModuleConf,
    SemiAdultDataModuleConf,
    SimpleXDataModuleConf,
    ThirdWayDataModuleConf,
)
from paf.config_classes.pytorch_lightning.trainer.configs import (  # type: ignore[import]
    TrainerConf,
)
from paf.log_progress import do_log
from paf.mmd import KernelType, mmd2
from paf.plotting import label_plot, make_data_plots
from paf.scoring import get_full_breakdown, produce_baselines
from paf.selection import baseline_selection_rules, produce_selection_groups
import wandb

LOGGER = logging.getLogger(__name__)
sns.set_theme(style="whitegrid")


class ModelType(Enum):
    PAF = auto()
    NN = auto()
    ERM_DP = auto()
    EQ_DP = auto()
    ERM_KAM = auto()
    EQ_KAM = auto()
    ERM_SHI = auto()
    EQ_SHI = auto()
    ERM_ZAF = auto()
    EQ_ZAF = auto()


@dataclass
class ExpConfig:
    """Experiment config."""

    lr: float = 1.0e-3
    weight_decay: float = 1.0e-3
    momentum: float = 0.9
    seed: int = 42
    log_offline: Optional[bool] = False
    tags: str = ""
    model: ModelType = ModelType.PAF
    debug: bool = False
    constrained: Optional[List[str]] = None


@dataclass
class Config:
    """Base Config Schema."""

    _target_: str = "paf.main.Config"
    data: Any = MISSING
    enc: Any = MISSING  # put config files for this into `conf/model/`
    clf: Any = MISSING  # put config files for this into `conf/model/`
    enc_trainer: Any = MISSING
    clf_trainer: Any = MISSING
    exp: ExpConfig = MISSING
    exp_group: Optional[str] = None


warnings.simplefilter(action="ignore", category=RuntimeWarning)

CS = ConfigStore.instance()
CS.store(name="config_schema", node=Config)  # General Schema
CS.store(name="enc_trainer_schema", node=TrainerConf, package="enc_trainer")
CS.store(name="clf_trainer_schema", node=TrainerConf, package="clf_trainer")

CLF_PKG: Final[str] = "clf"
CLF_GROUP: Final[str] = "schema/clf"
CS.store(name="clf_schema", node=ClfConf, package=CLF_PKG, group=CLF_GROUP)
CS.store(name="agarwal", node=AgarwalConf, package=CLF_PKG, group=CLF_GROUP)
CS.store(name="dp_oracle", node=DPOracleConf, package=CLF_PKG, group=CLF_GROUP)
CS.store(name="kamiran", node=KamiranConf, package=CLF_PKG, group=CLF_GROUP)
CS.store(name="kamishima", node=KamishimaConf, package=CLF_PKG, group=CLF_GROUP)
CS.store(name="lrcv", node=LRCVConf, package=CLF_PKG, group=CLF_GROUP)
CS.store(name="oracle", node=OracleConf, package=CLF_PKG, group=CLF_GROUP)
CS.store(name="zafar", node=ZafarFairnessConf, package=CLF_PKG, group=CLF_GROUP)

ENC_PKG: Final[str] = "enc"
ENC_GROUP: Final[str] = "schema/enc"
CS.store(name="enc_schema", node=AEConf, package=ENC_PKG, group=ENC_GROUP)
CS.store(name="cyclegan", node=CycleGanConf, package=ENC_PKG, group=ENC_GROUP)
CS.store(name="nearestneighbour", node=NearestNeighbourConf, package=ENC_PKG, group=ENC_GROUP)

DATA_PKG: Final[str] = "data"  # package:dir_within_config_path
DATA_GROUP: Final[str] = "schema/data"  # group
CS.store(name="adult-bolt", node=AdultDataModuleConf, package=DATA_PKG, group=DATA_GROUP)
CS.store(name="admiss-bolt", node=AdmissionsDataModuleConf, package=DATA_PKG, group=DATA_GROUP)
CS.store(name="compas-bolt", node=CompasDataModuleConf, package=DATA_PKG, group=DATA_GROUP)
CS.store(name="credit-bolt", node=CreditDataModuleConf, package=DATA_PKG, group=DATA_GROUP)
CS.store(name="crime-bolt", node=CrimeDataModuleConf, package=DATA_PKG, group=DATA_GROUP)
CS.store(name="health-bolt", node=HealthDataModuleConf, package=DATA_PKG, group=DATA_GROUP)
CS.store(name="law-bolt", node=LawDataModuleConf, package=DATA_PKG, group=DATA_GROUP)
CS.store(name="semi-synth", node=SemiAdultDataModuleConf, package=DATA_PKG, group=DATA_GROUP)
CS.store(name="sqf-bolt", node=SqfDataModuleConf, package=DATA_PKG, group=DATA_GROUP)
CS.store(name="lilliput", node=LilliputDataModuleConf, package=DATA_PKG, group=DATA_GROUP)
CS.store(name="synth", node=SimpleXDataModuleConf, package=DATA_PKG, group=DATA_GROUP)
CS.store(name="third", node=ThirdWayDataModuleConf, package=DATA_PKG, group=DATA_GROUP)


PS: Final[str] = "PostSelection"
TL: Final[str] = "Truelabels"
RW: Final[str] = "RealWorld"
EVAL_KERNEL: Final[KernelType] = KernelType.LINEAR


@hydra.main(config_path="configs", config_name="base_conf")
def launcher(hydra_config: DictConfig) -> None:
    """Instantiate with hydra and get the experiments running!"""
    cfg: Config = instantiate(hydra_config, _recursive_=True, _convert_="partial")
    run_paf(cfg, raw_config=OmegaConf.to_container(hydra_config, resolve=True, enum_to_str=True))


def name_lookup(cfg: Config) -> str:
    if isinstance(cfg.clf, em.InAlgorithm):
        return cfg.clf.name
    else:
        return cfg.exp.model.name + cfg.enc.name


def run_paf(cfg: Config, raw_config: Any) -> None:
    """Run the X Autoencoder."""
    pl.seed_everything(cfg.exp.seed, workers=True)
    data: BaseDataModule = cfg.data
    data.prepare_data()
    data.setup()

    indices = (
        [
            i
            for i, col in enumerate(data.test_datatuple.x.columns)
            for con in cfg.exp.constrained
            if con in col
        ]
        if cfg.exp.constrained is not None
        else []
    )

    LOGGER.info(f"data_dim={data.size()}, num_s={data.card_s}")

    raw_config["name"] = name_lookup(cfg)
    raw_config["data_name"] = cfg.data.__class__.__name__

    wandb_logger = pll.WandbLogger(
        entity="predictive-analytics-lab",
        project=f"paf_journal_{cfg.exp_group}",
        tags=cfg.exp.tags.split("/")[:-1],
        config=raw_config,
        offline=cfg.exp.log_offline,
    )
    cfg.enc_trainer.logger = wandb_logger
    cfg.clf_trainer.logger = wandb_logger

    pred_trainer = copy(cfg.enc_trainer)

    if cfg.exp.debug:
        make_data_plots(data, wandb_logger)

    if isinstance(cfg.clf, em.InAlgorithm):
        baseline_models(cfg.clf, data=data, logger=wandb_logger, debug=cfg.exp.debug)
        return

    if cfg.exp.model in (
        ModelType.ERM_DP,
        ModelType.EQ_DP,
        ModelType.ERM_SHI,
        ModelType.EQ_SHI,
        ModelType.ERM_ZAF,
        ModelType.EQ_ZAF,
        ModelType.ERM_KAM,
        ModelType.EQ_KAM,
    ):
        two_model_approach(cfg=cfg, data=data, logger=wandb_logger)
        return

    encoder = cfg.enc
    assert encoder is not None
    encoder.build(
        num_s=data.card_s,
        data_dim=data.size()[0],
        s_dim=data.dim_s[0],
        cf_available=data.cf_available if hasattr(data, "cf_available") else False,
        feature_groups=data.feature_groups,
        outcome_cols=data.disc_features + data.cont_features,
        indices=indices,
        data=data,
    )

    cfg.enc_trainer.callbacks += [
        L1Logger(),
        # MmdLogger(),
        # FeaturePlots()
    ]
    cfg.enc_trainer.fit(
        model=encoder,
        train_dataloaders=data.train_dataloader(shuffle=True, drop_last=True),
        val_dataloaders=data.val_dataloader(),
    )
    cfg.enc_trainer.test(dataloaders=data.test_dataloader(), ckpt_path=None)

    classifier = cfg.clf
    classifier.build(
        num_s=data.card_s,
        data_dim=data.size()[0],
        s_dim=data.dim_s[0],
        cf_available=data.cf_available if hasattr(data, "cf_available") else False,
        feature_groups=data.feature_groups,
        outcome_cols=data.disc_features + data.cont_features,
        scaler=None,
    )
    cfg.clf_trainer.fit(
        model=classifier,
        train_dataloaders=data.train_dataloader(shuffle=True, drop_last=True),
        val_dataloaders=data.val_dataloader(),
    )
    cfg.clf_trainer.test(dataloaders=data.test_dataloader(), ckpt_path=None)

    model = PafModel(encoder=encoder, classifier=classifier)

    # cfg.enc_trainer.fit(model=model, datamodule=data)
    results = model.collate_results(
        cfg.enc_trainer.predict(model=model, dataloaders=data.test_dataloader(), ckpt_path=None),
        cycle_steps=100,
    )

    if isinstance(results, PafResults):
        wandb_logger.experiment.log(results.cyc_vals.mean(axis="rows").to_dict())

        if cfg.exp.debug:
            _s = data.test_datatuple.s.copy().to_numpy()
            _y = data.test_datatuple.y.copy().to_numpy()

            for arr, name in [
                (results.x.detach().cpu().numpy(), "Input"),
                (results.enc_z.detach().cpu().numpy(), "Enc Embedding"),
                (results.recon.detach().cpu().numpy(), "Enc Recon"),
                (results.clf_z0.detach().cpu().numpy(), "Clf Xs=0 Embedding"),
                (results.clf_z1.detach().cpu().numpy(), "Clf Xs=1 Embedding"),
            ]:
                make_umap(
                    arr,
                    data_name=name,
                    cfg=cfg,
                    s=_s,
                    y=_y,
                    logger=wandb_logger,
                )

    evaluate(
        cfg=cfg,
        results=results,
        wandb_logger=wandb_logger,
        data=data,
        encoder=encoder,
        classifier=classifier,
        _model_trainer=pred_trainer,
    )

    if not cfg.exp.log_offline and wandb_logger is not None:
        wandb_logger.experiment.finish()


def make_umap(
    data: np.ndarray,
    data_name: str,
    cfg: Config,
    s: np.ndarray,
    y: np.ndarray,
    logger: pll.WandbLogger,
) -> None:
    reducer = umap.UMAP(random_state=cfg.exp.seed)
    scaled_embedding = StandardScaler().fit_transform(data)
    embedding = pd.DataFrame(reducer.fit_transform(scaled_embedding), columns=["x1", "x2"])
    embedding["s"] = s.copy()
    embedding["y"] = y.copy()

    if len(np.unique(s)) > 2:
        print(embedding["s"])
    if len(np.unique(y)) > 2:
        print(embedding["y"])

    conditions = [
        (embedding["s"] == 0) & (embedding["y"] == 0),
        (embedding["s"] == 0) & (embedding["y"] == 1),
        (embedding["s"] == 1) & (embedding["y"] == 0),
        (embedding["s"] == 1) & (embedding["y"] == 1),
    ]
    values = ["s0y0", "s0y1", "s1y0", "s1y1"]
    embedding["group"] = np.select(conditions, values, -1)

    sns.scatterplot(data=embedding, x="x1", y="x2", hue="group")
    logger.experiment.log({f"{data_name}": wandb.Image(plt)})
    plt.clf()


def evaluate(
    cfg: Config,
    *,
    results: Results,
    wandb_logger: pll.WandbLogger,
    data: BaseDataModule,
    encoder: pl.LightningModule,
    classifier: pl.LightningModule,
    _model_trainer: pl.Trainer,
) -> None:

    baseline_mmd = mmd2(results.x.clone(), results.x.clone(), kernel=EVAL_KERNEL)
    x2cf_mmd = mmd2(results.recon.clone(), results.cf_x.clone(), kernel=EVAL_KERNEL)
    recon_mmd = mmd2(results.recon.clone(), results.x.clone(), kernel=EVAL_KERNEL)
    s0_dist_mmd = mmd2(
        results.x[results.s == 0].clone(), results.recons_0.clone(), kernel=EVAL_KERNEL, biased=True
    )
    s1_dist_mmd = mmd2(
        results.x[results.s == 1].clone(), results.recons_1.clone(), kernel=EVAL_KERNEL, biased=True
    )

    for title, val in [
        ("MMD X vs X", baseline_mmd.item()),
        ("MMD X vs Cf", x2cf_mmd.item()),
        ("MMD X vs X^", recon_mmd.item()),
        ("MMD S0 vs Cf", s0_dist_mmd.item()),
        ("MMD S1 vs Cf", s1_dist_mmd.item()),
        ("P(Y=1|Sx=0,Sy=0", results.pd_results["s1_0_s2_0"].mean()),
        ("P(Y=1|Sx=0,Sy=1", results.pd_results["s1_0_s2_1"].mean()),
        ("P(Y=1|Sx=1,Sy=0", results.pd_results["s1_1_s2_0"].mean()),
        ("P(Y=1|Sx=1,Sy=1", results.pd_results["s1_1_s2_1"].mean()),
    ]:
        do_log(
            name="Logging/" + title,
            val=round(val, 5),
            logger=wandb_logger,
        )

    for fair_bool in (True, False):
        if cfg.exp.model == ModelType.PAF:
            preds = produce_selection_groups(
                outcomes=results.pd_results,
                data=data,
                recon_0=results.recons_0,
                recon_1=results.recons_1,
                logger=wandb_logger,
                data_name="Outcomes",
                fair=fair_bool,
                debug=cfg.exp.debug,
            )
        else:
            preds = baseline_selection_rules(
                outcomes=results.pd_results,
                logger=wandb_logger,
                fair=fair_bool,
                data_name="Outcomes",
            )
        metrics_and_breakdown(
            preds=preds,
            target=data.test_datatuple,
            name=f"{PS}/{fair_bool=}",
            logger=wandb_logger,
            debug=cfg.exp.debug,
            x=data.test_datatuple.x.copy(),
            s=data.test_datatuple.s.copy(),
            y=preds.hard.to_frame().copy(),
            graduated=None,
        )
        if isinstance(data, BaseDataModule) and data.cf_available:
            assert data.true_data_group is not None

            metrics_and_breakdown(
                preds=preds,
                target=data.true_test_datatuple,
                name=f"{PS}/{TL}/{fair_bool=}",
                logger=wandb_logger,
                debug=cfg.exp.debug,
                x=data.test_datatuple.x.copy(),
                s=data.test_datatuple.s.copy(),
                y=preds.hard.to_frame().copy(),
                graduated=data.true_test_datatuple,
            )

    if cfg.exp.model == ModelType.PAF:
        our_clf_preds = em.Prediction(
            hard=pd.Series(results.preds.squeeze(-1).detach().cpu().numpy())
        )
        metrics_and_breakdown(
            preds=our_clf_preds,
            target=data.test_datatuple,
            name=f"{RW}",
            logger=wandb_logger,
            debug=cfg.exp.debug,
            x=data.test_datatuple.x.copy(),
            s=data.test_datatuple.s.copy(),
            y=our_clf_preds.hard.to_frame().copy(),
            graduated=None,
        )
        if isinstance(data, BaseDataModule) and data.cf_available:
            assert data.true_data_group is not None
            metrics_and_breakdown(
                preds=our_clf_preds,
                target=data.true_test_datatuple,
                name=f"{RW}/{TL}",
                logger=wandb_logger,
                debug=cfg.exp.debug,
                x=data.test_datatuple.x.copy(),
                s=data.test_datatuple.s.copy(),
                y=our_clf_preds.hard.to_frame().copy(),
                graduated=data.true_test_datatuple,
            )
        if isinstance(cfg.enc, AE) and cfg.enc_trainer.max_epochs > 1:
            produce_baselines(
                encoder=encoder,
                datamodule=data,
                logger=wandb_logger,
                test_mode=cfg.enc_trainer.fast_dev_run,
            )
            produce_baselines(
                encoder=classifier,
                datamodule=data,
                logger=wandb_logger,
                test_mode=cfg.clf_trainer.fast_dev_run,
            )

        # # === This is only for reporting ====
        # _model = PafModel(encoder=encoder, classifier=classifier)
        # _results = _model.collate_results(
        #     _model_trainer.predict(
        #         model=_model, ckpt_path=None, dataloaders=data.train_dataloader()
        #     )
        # )
        # produce_selection_groups(
        #     outcomes=_results.pd_results,
        #     data=data,
        #     recon_0=_results.recons_0,
        #     recon_1=_results.recons_1,
        #     logger=wandb_logger,
        #     data_name="Train",
        #     debug=cfg.exp.debug,
        # )
        # # === === ===


def baseline_models(
    model: em.InAlgorithm, *, data: BaseDataModule, logger: pll.WandbLogger, debug: bool
) -> None:
    LOGGER.info(f"=== {model.name} ===")
    results = model.run(data.train_datatuple, data.test_datatuple)
    metrics_and_breakdown(
        preds=results,
        target=data.test_datatuple,
        name=f"{RW}",
        logger=logger,
        debug=debug,
        x=data.test_datatuple.x.copy(),
        s=data.test_datatuple.s.copy(),
        y=results.hard.to_frame().copy(),
        graduated=None,
    )

    if isinstance(data, BaseDataModule):
        LOGGER.info(f"=== {model.name} and 'True' Data ===")
        results = model.run(data.train_datatuple, data.test_datatuple)
        assert data.true_test_datatuple is not None

        metrics_and_breakdown(
            preds=results,
            target=data.true_test_datatuple,
            name=f"{RW}/{TL}",
            logger=logger,
            debug=debug,
            x=data.test_datatuple.x.copy(),
            s=data.test_datatuple.s.copy(),
            y=results.hard.to_frame().copy(),
            graduated=data.true_test_datatuple,
        )


def multiple_metrics(
    preds: em.Prediction, *, target: em.DataTuple, name: str, logger: pll.WandbLogger, debug: bool
) -> None:
    """Get multiple metrics."""
    if debug:
        try:
            label_plot(
                em.DataTuple(x=target.x.copy(), s=target.s.copy(), y=preds.hard.to_frame()),
                logger,
                name,
            )
        except (IndexError, KeyError):
            pass

    results = em.run_metrics(
        predictions=preds,
        actual=target,
        metrics=[em.Accuracy(), em.ProbPos()],
        per_sens_metrics=[em.Accuracy(), em.ProbPos(), em.TPR(), em.TNR()],
        use_sens_name=False,
    )
    for key, value in results.items():
        do_log(f"{name}/{key.replace('/', '%')}", value, logger)


def two_model_approach(cfg: Config, data: BaseDataModule, logger: pll.WandbLogger) -> None:
    if cfg.exp.model in (ModelType.ERM_DP, ModelType.ERM_KAM, ModelType.ERM_ZAF, ModelType.ERM_SHI):
        first_model = em.LRCV(seed=cfg.exp.seed)
        first_results = first_model.run(data.train_datatuple, data.test_datatuple)
    else:
        erm_model = em.LRCV(seed=cfg.exp.seed)
        first_model = em.Hardt(seed=cfg.exp.seed)
        first_results = first_model.run(
            train_predictions=erm_model.run(data.train_datatuple, data.train_datatuple),
            train=data.train_datatuple,
            test_predictions=erm_model.run(data.train_datatuple, data.test_datatuple),
            test=data.test_datatuple,
        )
    if cfg.exp.model in (ModelType.ERM_DP, ModelType.EQ_DP):
        dp_model = em.Agarwal(fairness="DP", seed=cfg.exp.seed)
        dp_results = dp_model.run(data.train_datatuple, data.test_datatuple)
    elif cfg.exp.model in (ModelType.ERM_ZAF, ModelType.EQ_ZAF):
        dp_model = em.ZafarFairness()
        dp_results = dp_model.run(data.train_datatuple, data.test_datatuple)
    elif cfg.exp.model in (ModelType.ERM_KAM, ModelType.EQ_KAM):
        dp_model = em.Kamiran(seed=cfg.exp.seed)
        dp_results = dp_model.run(data.train_datatuple, data.test_datatuple)
    elif cfg.exp.model in (ModelType.ERM_SHI, ModelType.EQ_SHI):
        dp_model = em.Kamishima()
        dp_results = dp_model.run(data.train_datatuple, data.test_datatuple)
    else:
        raise NotImplementedError(f"{cfg.exp.model} not implemented")

    matches = {
        f"{c}": first_results.hard[
            pd.concat([dp_results.hard, first_results.hard], axis=1).sum(axis=1) == c
        ].count()
        for c in range(3)
    }
    print(matches)
    df = pd.DataFrame.from_dict(
        {
            "s1_0_s2_0": first_results.hard.values,
            "s1_1_s2_1": dp_results.hard.values,
            "true_s": data.test_datatuple.s.copy().values[:, 0],
        }
    )

    for fair_bool in (True, False):
        preds = baseline_selection_rules(
            outcomes=df,
            logger=logger,
            fair=fair_bool,
            data_name="Outcomes",
        )
        metrics_and_breakdown(
            preds=preds,
            target=data.test_datatuple,
            name=f"{PS}/{fair_bool=}",
            logger=logger,
            debug=cfg.exp.debug,
            x=data.test_datatuple.x.copy(),
            s=data.test_datatuple.s.copy(),
            y=preds.hard.to_frame().copy(),
            graduated=None,
        )
        if isinstance(data, BaseDataModule) and data.cf_available:
            assert data.true_data_group is not None
            metrics_and_breakdown(
                preds=preds,
                target=data.test_datatuple,
                name=f"{PS}/{TL}/{fair_bool=}",
                logger=logger,
                debug=cfg.exp.debug,
                x=data.test_datatuple.x.copy(),
                s=data.test_datatuple.s.copy(),
                y=preds.hard.to_frame().copy(),
                graduated=data.true_test_datatuple,
            )

    metrics_and_breakdown(
        preds=first_results,
        target=data.test_datatuple,
        name=f"{RW}",
        logger=logger,
        debug=cfg.exp.debug,
        x=data.test_datatuple.x.copy(),
        s=data.test_datatuple.s.copy(),
        y=first_results.hard.to_frame().copy(),
        graduated=None,
    )
    if isinstance(data, BaseDataModule) and data.cf_available:
        assert data.true_data_group is not None
        metrics_and_breakdown(
            preds=first_results,
            target=data.true_test_datatuple,
            name=f"{RW}/{TL}",
            logger=logger,
            debug=cfg.exp.debug,
            x=data.test_datatuple.x.copy(),
            s=data.test_datatuple.s.copy(),
            y=first_results.hard.to_frame().copy(),
            graduated=data.true_test_datatuple,
        )


def metrics_and_breakdown(
    preds: em.Prediction,
    *,
    target: em.DataTuple,
    name: str,
    logger: pll.WandbLogger,
    debug: bool,
    x: pd.DataFrame,
    s: pd.DataFrame,
    y: pd.DataFrame,
    graduated: em.DataTuple | None,
) -> None:
    multiple_metrics(preds=preds, target=target, name=name, logger=logger, debug=debug)
    get_full_breakdown(
        target_info="Stats/" + name,
        acceptance=em.DataTuple(x=x, s=s, y=y),
        graduated=graduated,
        logger=logger,
    )


if __name__ == "__main__":
    launcher()
