from __future__ import annotations
from typing import Final

from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf
import pytest

from paf.main import Config, run_aies

CFG_PTH: Final[str] = "../paf/configs"
SCHEMAS: Final[list[str]] = [
    "enc=basic",
    "clf=basic",
    "exp=unit_test",
    "trainer=unit_test",
]


@pytest.mark.parametrize("dm_schema", ["ad"])
def test_enc(dm_schema: str) -> None:
    """Test the encoder network runs."""
    with initialize(config_path=CFG_PTH):
        # config is relative to a module
        hydra_cfg = compose(
            config_name="base_conf",
            overrides=[f"data={dm_schema}"] + SCHEMAS,
        )
        cfg: Config = instantiate(hydra_cfg, _recursive_=True, _convert_="partial")

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
            scaler=cfg.data.scaler,
        )
        cfg.trainer.fit(model=encoder, datamodule=cfg.data)
        cfg.trainer.test(model=encoder, ckpt_path=None, datamodule=cfg.data)


def test_nn() -> None:
    """Quick run on models to check nothing's broken."""
    with initialize(config_path=CFG_PTH):
        # config is relative to a module
        hydra_cfg = compose(
            config_name="base_conf",
            overrides=[f"data=lill"] + SCHEMAS + [f"exp.model=paf"],
        )
        cfg: Config = instantiate(hydra_cfg, _recursive_=True, _convert_="partial")
        run_aies(cfg, raw_config=OmegaConf.to_container(hydra_cfg, resolve=True, enum_to_str=True))


def test_cyclegan() -> None:
    """Quick run on models to check nothing's broken."""
    with initialize(config_path=CFG_PTH):
        # config is relative to a module
        hydra_cfg = compose(
            config_name="base_conf",
            overrides=[f"data=adm"] + SCHEMAS + [f"enc=cyc"],
        )
        cfg: Config = instantiate(hydra_cfg, _recursive_=True, _convert_="partial")
        run_aies(cfg, raw_config=OmegaConf.to_container(hydra_cfg, resolve=True, enum_to_str=True))
