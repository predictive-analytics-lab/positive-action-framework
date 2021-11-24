from __future__ import annotations
from typing import Final

if 1:
    import faiss  # noqa
import ethicml as em
import torch
from conduit.fair.data import AdultDataModule
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf
import pytest

from paf.architectures.model.nearestneighbour import KnnExact
from paf.main import Config, run_paf

CFG_PTH: Final[str] = "../paf/configs"
SCHEMAS: Final[list[str]] = [
    "enc=lill_1",
    "clf=lill_1",
    "exp=unit_test",
    "enc_trainer=unit_test",
    "clf_trainer=unit_test",
    "data=lill",
    "exp.constrained=[potions]",
]


def test_paf() -> None:
    with initialize(config_path=CFG_PTH):
        # config is relative to a module
        hydra_cfg = compose(
            config_name="base_conf",
            overrides=SCHEMAS + ["data=ad"],
        )

        cfg: Config = instantiate(hydra_cfg, _recursive_=True, _convert_="partial")
        run_paf(cfg, raw_config=OmegaConf.to_container(hydra_cfg, resolve=True, enum_to_str=True))


@pytest.mark.parametrize("dm_schema", ["ad", "law", "lill"])
def test_enc(dm_schema: str) -> None:
    """Test the encoder network runs."""
    with initialize(config_path=CFG_PTH):
        # config is relative to a module
        hydra_cfg = compose(
            config_name="base_conf",
            overrides=SCHEMAS + [f"data={dm_schema}"],
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
            indices=None,
            data=cfg.data,
        )
        cfg.enc_trainer.fit(model=encoder, datamodule=cfg.data)
        cfg.enc_trainer.test(model=encoder, ckpt_path=None, datamodule=cfg.data)


@pytest.mark.parametrize("dm_schema", ["ad", "law", "lill"])
def test_clf(dm_schema: str) -> None:
    """Test the encoder network runs."""
    with initialize(config_path=CFG_PTH):
        # config is relative to a module
        hydra_cfg = compose(
            config_name="base_conf",
            overrides=SCHEMAS + [f"data={dm_schema}"],
        )
        cfg: Config = instantiate(hydra_cfg, _recursive_=True, _convert_="partial")

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
            scaler=cfg.data.scaler,
        )
        cfg.clf_trainer.fit(model=classifier, datamodule=cfg.data)
        cfg.clf_trainer.test(model=classifier, ckpt_path=None, datamodule=cfg.data)


# @pytest.mark.parametrize("dm_schema", ["ad", "law", "lill"])
# def test_nn(dm_schema: str) -> None:
#     """Quick run on models to check nothing's broken."""
#     with initialize(config_path=CFG_PTH):
#         # config is relative to a module
# hydra_cfg = compose(
#     config_name="base_conf",
#     overrides=SCHEMAS + ["exp.model=NN", f"data={dm_schema}"],
# )
#
# cfg: Config = instantiate(hydra_cfg, _recursive_=True, _convert_="partial")
# run_paf(cfg, raw_config=OmegaConf.to_container(hydra_cfg, resolve=True, enum_to_str=True))


def test_knn() -> None:
    data = AdultDataModule(test_prop=0.7)
    data.prepare_data()
    data.setup()

    train_features = torch.as_tensor(data.train_datatuple.x.values, dtype=torch.float32)
    train_sens = torch.as_tensor(data.train_datatuple.s.values, dtype=torch.long)

    knn = KnnExact(k=1, normalize=False)

    batch = next(iter(data.train_dataloader()))
    x = batch.x
    s = batch.s

    features = torch.empty_like(x)
    sens = torch.empty_like(s, dtype=torch.long)
    for s_val in range(2):
        mask = (train_sens != s_val).squeeze()
        mask_inds = mask.nonzero(as_tuple=False).squeeze()
        knn_inds = knn(x=x[(s_val == s).squeeze()], y=train_features[mask_inds])
        abs_indices = mask_inds[knn_inds].squeeze()
        features[(s_val == s).squeeze()] = train_features[abs_indices]
        sens[(s_val == s).squeeze()] = train_sens[abs_indices].squeeze()

    with pytest.raises(AssertionError):
        assert (sens == s).all()
        assert torch.allclose(x, features)


@pytest.mark.parametrize("model", ["ERM_DP", "EQ_DP", "ERM_KAM", "EQ_KAM"])
@pytest.mark.parametrize("dm_schema", ["ad", "law", "lill"])
def test_erm_dp(model: str, dm_schema: str) -> None:
    """Quick run on models to check nothing's broken."""
    with initialize(config_path=CFG_PTH):
        # config is relative to a module
        hydra_cfg = compose(
            config_name="base_conf",
            overrides=SCHEMAS
            + [f"exp.model={model}", f"data={dm_schema}", "clf_trainer.max_epochs=1"],
        )

        cfg: Config = instantiate(hydra_cfg, _recursive_=True, _convert_="partial")
        run_paf(cfg, raw_config=OmegaConf.to_container(hydra_cfg, resolve=True, enum_to_str=True))


@pytest.mark.parametrize("dm_schema", ["ad", "law", "lill"])
@pytest.mark.parametrize("model", ["cyc", "nn"])
def test_model_type(model: str, dm_schema: str) -> None:
    """Quick run on models to check nothing's broken."""
    with initialize(config_path=CFG_PTH):
        # config is relative to a module
        hydra_cfg = compose(
            config_name="base_conf",
            overrides=SCHEMAS + [f"enc={model}", f"data={dm_schema}"],
        )

        cfg: Config = instantiate(hydra_cfg, _recursive_=True, _convert_="partial")
        run_paf(cfg, raw_config=OmegaConf.to_container(hydra_cfg, resolve=True, enum_to_str=True))


@pytest.mark.parametrize(
    "model", ["agarwal", "dp_oracle", "kamiran", "kamishima", "lrcv", "oracle", "zafar"]
)
def test_baselines(model: em.InAlgorithm) -> None:
    with initialize(config_path=CFG_PTH):
        hydra_cfg = compose(
            config_name="base_conf",
            overrides=SCHEMAS + [f"clf={model}"],
        )

        cfg: Config = instantiate(hydra_cfg, _recursive_=True, _convert_="partial")
        run_paf(cfg, raw_config=OmegaConf.to_container(hydra_cfg, resolve=True, enum_to_str=True))
