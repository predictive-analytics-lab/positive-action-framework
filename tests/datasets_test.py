"""Basic tests."""
import copy
from typing import Final, List

import pytest
import torch
from hydra.core.config_store import ConfigStore
from hydra.experimental import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

from paf.config_classes.paf.data_modules.configs import (
    LilliputDataModuleConf,
    SimpleAdultDataModuleConf,
    SimpleXDataModuleConf,
    ThirdWayDataModuleConf,
)
from paf.config_classes.pytorch_lightning.trainer.configs import TrainerConf
from paf.main import Config, data_group, data_package, run_aies
from paf.model.aies_model import AiesModel

cs = ConfigStore.instance()
cs.store(name="config_schema", node=Config)  # General Schema
cs.store(name="trainer_schema", node=TrainerConf, package="trainer")

cs.store(name="adult", node=SimpleAdultDataModuleConf, package=data_package, group=data_group)
cs.store(name="lilliput", node=LilliputDataModuleConf, package=data_package, group=data_group)
cs.store(name="synth", node=SimpleXDataModuleConf, package=data_package, group=data_group)
cs.store(name="third", node=ThirdWayDataModuleConf, package=data_package, group=data_group)

CFG_PTH: Final[str] = "../paf/configs"
SCHEMAS: Final[List[str]] = [
    "enc=basic",
    "clf=basic",
    "exp=unit_test",
    "trainer=unit_test",
]


@pytest.mark.parametrize("dm_schema", ["semi", "lill", "synth", "adult"])
def test_with_initialize(dm_schema: str) -> None:
    """Quick run on models to check nothing's broken."""
    with initialize(config_path=CFG_PTH):
        # config is relative to a module
        hydra_cfg = compose(
            config_name="base_conf",
            overrides=[f"data={dm_schema}"] + SCHEMAS,
        )
        cfg: Config = instantiate(hydra_cfg, _recursive_=True, _convert_="partial")
        run_aies(cfg, raw_config=OmegaConf.to_container(hydra_cfg, resolve=True, enum_to_str=True))


@pytest.mark.parametrize(
    "dm_schema,cf_available", [("third", True), ("lill", True), ("synth", True), ("adult", False)]
)
def test_data(dm_schema, cf_available):
    """Test the data module."""
    with initialize(config_path="../paf/configs"):
        # config is relative to a module
        hydra_cfg = compose(
            config_name="base_conf",
            overrides=[f"data={dm_schema}"] + SCHEMAS,
        )
        cfg: Config = instantiate(hydra_cfg, _recursive_=True, _convert_="partial")
        seed_everything(0)

        cfg.data.prepare_data()

        assert cf_available == cfg.data.cf_available

        for batch in cfg.data.train_dataloader():
            if cfg.data.cf_available:
                x, s, y, cf_x, cf_s, cf_y, _ = batch
                with pytest.raises(AssertionError):
                    torch.testing.assert_allclose(x, cf_x)
                with pytest.raises(AssertionError):
                    torch.testing.assert_allclose(s, cf_s)
                with pytest.raises(AssertionError):
                    torch.testing.assert_allclose(y, cf_y)
            else:
                x, s, y, _ = batch
                torch.testing.assert_allclose(x, x)
                torch.testing.assert_allclose(s, s)
                torch.testing.assert_allclose(y, y)


@pytest.mark.parametrize(
    "dm_schema,cf_available", [("third", True), ("lill", True), ("synth", True), ("adult", False)]
)
def test_data(dm_schema, cf_available):
    """Test the flip dataset function."""
    with initialize(config_path=CFG_PTH):
        # config is relative to a module
        hydra_cfg = compose(
            config_name="base_conf",
            overrides=[f"data={dm_schema}"] + SCHEMAS,
        )
        cfg: Config = instantiate(hydra_cfg, _recursive_=True, _convert_="partial")
        seed_everything(0)
        data = cfg.data
        data.prepare_data()

        training_dl = data.train_dataloader(shuffle=False, drop_last=False)
        test_dl = data.test_dataloader(shuffle=False, drop_last=False)
        data.flip_train_test()
        training_dl2 = data.train_dataloader(shuffle=False, drop_last=False)
        test_dl2 = data.test_dataloader(shuffle=False, drop_last=False)

        for (tr_batch, te_batch) in zip(training_dl, training_dl2):
            if data.cf_available:
                tr_x, tr_s, tr_y, _, _, _, _ = tr_batch
                te_x, te_s, te_y, _, _, _, _ = te_batch
            else:
                tr_x, tr_s, tr_y, _ = tr_batch
                te_x, te_s, te_y, _ = te_batch

            with pytest.raises(AssertionError):
                torch.testing.assert_allclose(tr_x, te_x)


@pytest.mark.parametrize("dm_schema", ["third", "lill", "synth", "adult"])
def test_enc(dm_schema):
    """Test the encoder network runs."""
    with initialize(config_path=CFG_PTH):
        # config is relative to a module
        hydra_cfg = compose(
            config_name="base_conf",
            overrides=[f"data={dm_schema}"] + SCHEMAS,
        )
        cfg: Config = instantiate(hydra_cfg, _recursive_=True, _convert_="partial")

        cfg.data.prepare_data()
        encoder = cfg.enc
        encoder.build(
            num_s=cfg.data.num_s,
            data_dim=cfg.data.data_dim,
            s_dim=cfg.data.s_dim,
            cf_available=cfg.data.cf_available,
            feature_groups=cfg.data.feature_groups,
            outcome_cols=cfg.data.column_names,
            scaler=cfg.data.scaler,
        )
        cfg.trainer.fit(model=encoder, datamodule=cfg.data)
        cfg.trainer.test(model=encoder, ckpt_path=None, datamodule=cfg.data)


@pytest.mark.parametrize("dm_schema", ["third", "lill", "synth", "adult"])
def test_clf(dm_schema):
    """Test the classifier network runs."""
    with initialize(config_path=CFG_PTH):
        # config is relative to a module
        hydra_cfg = compose(
            config_name="base_conf",
            overrides=[f"data={dm_schema}"] + SCHEMAS,
        )
        cfg: Config = instantiate(hydra_cfg, _recursive_=True, _convert_="partial")

        cfg.data.prepare_data()
        classifier = cfg.clf
        classifier.build(
            num_s=cfg.data.num_s,
            data_dim=cfg.data.data_dim,
            s_dim=cfg.data.s_dim,
            cf_available=cfg.data.cf_available,
            feature_groups=cfg.data.feature_groups,
            outcome_cols=cfg.data.column_names,
        )
        cfg.trainer.fit(model=classifier, datamodule=cfg.data)
        cfg.trainer.test(model=classifier, ckpt_path=None, datamodule=cfg.data)


@pytest.mark.parametrize("dm_schema", ["third", "lill", "synth", "adult"])
def test_clf(dm_schema):
    """Test the end to end."""
    with initialize(config_path=CFG_PTH):
        # config is relative to a module
        hydra_cfg = compose(
            config_name="base_conf",
            overrides=[f"data={dm_schema}"] + SCHEMAS,
        )
        cfg: Config = instantiate(hydra_cfg, _recursive_=True, _convert_="partial")

        enc_trainer = copy.deepcopy(cfg.trainer)
        clf_trainer = copy.deepcopy(cfg.trainer)
        model_trainer = copy.deepcopy(cfg.trainer)

        data = cfg.data
        data.prepare_data()
        encoder = cfg.enc
        encoder.build(
            num_s=data.num_s,
            data_dim=data.data_dim,
            s_dim=data.s_dim,
            cf_available=data.cf_available,
            feature_groups=data.feature_groups,
            outcome_cols=data.column_names,
            scaler=cfg.data.scaler,
        )
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
        clf_trainer.fit(model=classifier, datamodule=data)
        clf_trainer.test(model=classifier, ckpt_path=None, datamodule=data)

        model = AiesModel(encoder=encoder, classifier=classifier)
        model_trainer.fit(model=model, datamodule=data)
        model_trainer.test(model=model, ckpt_path=None, datamodule=data)

        print(model.pd_results)
