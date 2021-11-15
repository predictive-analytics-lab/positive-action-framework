"""Basic tests."""
from __future__ import annotations
import copy
from typing import Final

from hydra.core.config_store import ConfigStore
from hydra.experimental import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf
import pytest
import pytorch_lightning as pl
from sklearn.preprocessing import MinMaxScaler
import torch

from paf.architectures.paf_model import PafModel
from paf.config_classes.pytorch_lightning.trainer.configs import (  # type: ignore[import]
    TrainerConf,
)
from paf.main import Config, run_paf

cs = ConfigStore.instance()
cs.store(name="config_schema", node=Config)  # General Schema
cs.store(name="trainer_schema", node=TrainerConf, package="trainer")

CFG_PTH: Final[str] = "../paf/configs"
SCHEMAS: Final[list[str]] = [
    "enc=basic",
    "clf=basic",
    "exp=unit_test",
    "enc_trainer=unit_test",
    "clf_trainer=unit_test",
]


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
def test_get_value_counts(seed: int):
    with initialize(config_path="../paf/configs"):
        # config is relative to a module
        hydra_cfg = compose(
            config_name="base_conf",
            overrides=["data=lill", f"exp.seed={seed}"] + SCHEMAS,
        )
        cfg: Config = instantiate(hydra_cfg, _recursive_=True, _convert_="partial")
        pl.seed_everything(seed)

        cfg.data.prepare_data()
        cfg.data.setup()


@pytest.mark.parametrize(
    "dm_schema", ["ad", "adm", "adminv", "law", "semi", "lill", "synth"]  # "crime", "health"
)
def test_with_initialize(dm_schema: str) -> None:
    """Quick run on models to check nothing's broken."""
    with initialize(config_path=CFG_PTH):
        # config is relative to a module
        hydra_cfg = compose(
            config_name="base_conf",
            overrides=[f"data={dm_schema}"] + SCHEMAS,
        )
        cfg: Config = instantiate(hydra_cfg, _recursive_=True, _convert_="partial")
        run_paf(cfg, raw_config=OmegaConf.to_container(hydra_cfg, resolve=True, enum_to_str=True))


@pytest.mark.parametrize(
    "dm_schema,cf_available",
    [
        ("third", True),
        ("lill", True),
        ("synth", True),
        ("ad", False),
        ("adm", False),
        # ("crime", False),
        ("law", False),
        ("health", False),
    ],
)
def test_data(dm_schema: str, cf_available: bool) -> None:
    """Test the data module."""
    with initialize(config_path="../paf/configs"):
        # config is relative to a module
        hydra_cfg = compose(
            config_name="base_conf",
            overrides=[f"data={dm_schema}"] + SCHEMAS,
        )
        cfg: Config = instantiate(hydra_cfg, _recursive_=True, _convert_="partial")
        pl.seed_everything(0)

        cfg.data.prepare_data()
        cfg.data.setup()

        assert cf_available == (
            cfg.data.cf_available if hasattr(cfg.data, "cf_available") else False
        )

        assert cfg.data.card_s == 2
        assert cfg.data.card_y == 2

        batch = next(iter(cfg.data.train_dataloader()))
        if cf_available:
            with pytest.raises(AssertionError):
                torch.testing.assert_allclose(batch.x, batch.cfx)
            with pytest.raises(AssertionError):
                torch.testing.assert_allclose(batch.s, batch.cfs)
            with pytest.raises(AssertionError):
                torch.testing.assert_allclose(batch.y, batch.cfy)


@pytest.mark.parametrize("dm_schema", ["third", "lill", "synth", "ad"])
def test_datamods(dm_schema: str) -> None:
    """Test the flip dataset function."""
    with initialize(config_path=CFG_PTH):
        # config is relative to a module
        hydra_cfg = compose(
            config_name="base_conf",
            overrides=[f"data={dm_schema}"] + SCHEMAS,
        )
        cfg: Config = instantiate(hydra_cfg, _recursive_=True, _convert_="partial")
        pl.seed_everything(0)
        data = cfg.data
        data.prepare_data()
        cfg.data.setup()

        training_dl = data.train_dataloader(shuffle=False, drop_last=False)
        training_dl2 = data.test_dataloader()

        tr_batch, te_batch = next(zip(iter(training_dl), iter(training_dl2)))
        with pytest.raises(AssertionError):
            torch.testing.assert_allclose(tr_batch.x, te_batch.x)


@pytest.mark.parametrize("dm_schema", ["third", "lill", "synth", "adm", "ad"])
def test_enc(dm_schema: str) -> None:
    """Test the encoder network runs."""
    with initialize(config_path=CFG_PTH):
        # config is relative to a module
        hydra_cfg = compose(
            config_name="base_conf",
            overrides=[f"data={dm_schema}"] + SCHEMAS,
        )
        cfg: Config = instantiate(hydra_cfg, _recursive_=True, _convert_="partial")
        cfg.data.scaler = MinMaxScaler()

        cfg.data.prepare_data()
        cfg.data.setup()
        encoder = cfg.enc
        encoder.build(
            num_s=cfg.data.card_s,
            data_dim=cfg.data.size()[0],
            s_dim=cfg.data.dim_s[0],
            cf_available=cfg.data.cf_available if hasattr(cfg.data, "cf_available") else False,
            feature_groups=cfg.data.feature_groups,
            outcome_cols=cfg.data.disc_features + cfg.data.cont_features,
            indices=None,
            data=cfg.data,
        )
        cfg.enc_trainer.fit(model=encoder, datamodule=cfg.data)
        cfg.enc_trainer.test(model=encoder, ckpt_path=None, datamodule=cfg.data)


@pytest.mark.parametrize("dm_schema", ["third", "lill", "synth", "ad"])
def test_clf(dm_schema: str) -> None:
    """Test the classifier network runs."""
    with initialize(config_path=CFG_PTH):
        # config is relative to a module
        hydra_cfg = compose(
            config_name="base_conf",
            overrides=[f"data={dm_schema}"] + SCHEMAS,
        )
        cfg: Config = instantiate(hydra_cfg, _recursive_=True, _convert_="partial")
        cfg.data.scaler = MinMaxScaler()

        cfg.data.prepare_data()
        cfg.data.setup()
        classifier = cfg.clf
        classifier.build(
            num_s=cfg.data.card_s,
            data_dim=cfg.data.size()[0],
            s_dim=cfg.data.dim_s[0],
            cf_available=cfg.data.cf_available if hasattr(cfg.data, "cf_available") else False,
            feature_groups=cfg.data.feature_groups,
            outcome_cols=cfg.data.disc_features + cfg.data.cont_features,
            data=cfg.data,
        )
        cfg.clf_trainer.fit(model=classifier, datamodule=cfg.data)
        cfg.clf_trainer.test(model=classifier, ckpt_path=None, datamodule=cfg.data)


@pytest.mark.parametrize("dm_schema", ["third", "lill", "synth", "ad"])
def test_clfmod(dm_schema: str) -> None:
    """Test the end to end."""
    with initialize(config_path=CFG_PTH):
        # config is relative to a module
        hydra_cfg = compose(
            config_name="base_conf",
            overrides=[f"data={dm_schema}"] + SCHEMAS,
        )
        cfg: Config = instantiate(hydra_cfg, _recursive_=True, _convert_="partial")
        cfg.data.scaler = MinMaxScaler()

        model_trainer: pl.Trainer = copy.deepcopy(cfg.enc_trainer)

        data = cfg.data
        data.prepare_data()
        data.setup()
        encoder = cfg.enc
        encoder.build(
            num_s=data.card_s,
            data_dim=data.size()[0],
            s_dim=data.dim_s[0],
            cf_available=cfg.data.cf_available if hasattr(cfg.data, "cf_available") else False,
            feature_groups=data.feature_groups,
            outcome_cols=cfg.data.disc_features + cfg.data.cont_features,
            indices=None,
            data=data,
        )
        cfg.enc_trainer.fit(model=encoder, datamodule=data)
        cfg.enc_trainer.test(model=encoder, ckpt_path=None, datamodule=data)

        classifier = cfg.clf
        classifier.build(
            num_s=data.card_s,
            data_dim=data.size()[0],
            s_dim=data.dim_s[0],
            cf_available=cfg.data.cf_available if hasattr(cfg.data, "cf_available") else False,
            feature_groups=data.feature_groups,
            outcome_cols=cfg.data.disc_features + cfg.data.cont_features,
            data=data,
        )
        cfg.clf_trainer.fit(model=classifier, datamodule=data)
        cfg.clf_trainer.test(model=classifier, ckpt_path=None, datamodule=data)

        model = PafModel(encoder=encoder, classifier=classifier)
        model_trainer.fit(model=model, datamodule=data)
        results = model.collate_results(
            model_trainer.predict(model=model, ckpt_path=None, dataloaders=data.test_dataloader())
        )

        print(results.pd_results)
