"""Basic tests."""
import pytest
import torch
from pytorch_lightning import Trainer, seed_everything

from src.config_classes.dataclasses import (
    AdultConfig,
    DataConfig,
    KernelType,
    LilliputConfig,
    ModelConfig,
    SimpleXConfig,
    ThirdWayConfig,
)
from src.data_modules.create import create_data_module
from src.model.aies_model import AiesModel
from src.model.classifier_model import Clf
from src.model.encoder_model import AE

THIRD = ThirdWayConfig(
    batch_size=32,
    num_workers=1,
    seed=0,
    acceptance_rate=0.4,
    alpha=0.6,
    gamma=0.1,
    num_samples=100,
    num_features=5,
    beta=0.1,
    xi=0.01,
    num_hidden_features=30,
)

LILLIPUT = LilliputConfig(
    batch_size=32,
    num_workers=1,
    seed=0,
    alpha=0.6,
    gamma=0.1,
    num_samples=100,
)

SIMPLE = SimpleXConfig(
    batch_size=32,
    num_workers=1,
    seed=0,
    alpha=0.6,
    gamma=0.1,
    num_samples=100,
)

ADULT = AdultConfig(
    batch_size=32,
    num_workers=1,
    seed=0,
)

ENC = ModelConfig(
    blocks=0,
    latent_dims=5,
    latent_multiplier=1,
    lr=1e-3,
    adv_weight=1.0,
    target_weight=1.0,
    reg_weight=1e-3,
    s_as_input=True,
    mmd_kernel=KernelType.linear,
    scheduler_rate=0.999,
    weight_decay=1e-6,
)

CLF = ModelConfig(
    blocks=0,
    latent_dims=5,
    latent_multiplier=1,
    lr=1e-3,
    adv_weight=1.0,
    target_weight=1.0,
    reg_weight=1e-3,
    s_as_input=True,
    mmd_kernel=KernelType.linear,
    scheduler_rate=0.999,
    weight_decay=1e-6,
)


@pytest.mark.parametrize("cfg,cf_available", [(THIRD, True), (LILLIPUT, True), (SIMPLE, True), (ADULT, False)])
def test_data(cfg, cf_available):
    """Test the data module."""
    seed_everything(0)

    data = create_data_module(cfg)
    data.prepare_data()

    assert cf_available == data.cf_available

    for batch in data.train_dataloader():
        if data.cf_available:
            x, s, y, cf_x, cf_s, cf_y = batch
            with pytest.raises(AssertionError):
                torch.testing.assert_allclose(x, cf_x)
            with pytest.raises(AssertionError):
                torch.testing.assert_allclose(s, cf_s)
            with pytest.raises(AssertionError):
                torch.testing.assert_allclose(y, cf_y)
        else:
            x, s, y = batch
            torch.testing.assert_allclose(x, x)
            torch.testing.assert_allclose(s, s)
            torch.testing.assert_allclose(y, y)


def test_enc():
    """Test the encoder network runs."""
    data = create_data_module(SIMPLE)
    data.prepare_data()
    encoder = AE(
        ENC,
        num_s=data.num_s,
        data_dim=data.data_dim,
        s_dim=data.s_dim,
        cf_available=data.cf_available,
        feature_groups=data.feature_groups,
        column_names=data.column_names,
    )

    enc_trainer = Trainer(fast_dev_run=True, logger=False)
    enc_trainer.fit(encoder, datamodule=data)
    enc_trainer.test(ckpt_path=None, datamodule=data)


def test_clf():
    """Test the classifier network runs."""
    data = create_data_module(SIMPLE)
    data.prepare_data()

    classifier = Clf(
        CLF,
        num_s=data.num_s,
        data_dim=data.data_dim,
        s_dim=data.s_dim,
        cf_available=data.cf_available,
        outcome_cols=data.outcome_columns,
    )

    clf_trainer = Trainer(fast_dev_run=True, logger=False)
    clf_trainer.fit(classifier, datamodule=data)
    clf_trainer.test(ckpt_path=None, datamodule=data)


def test_all():
    """Test the end to end."""
    data = create_data_module(SIMPLE)
    data.prepare_data()
    encoder = AE(
        ENC,
        num_s=data.num_s,
        data_dim=data.data_dim,
        s_dim=data.s_dim,
        cf_available=data.cf_available,
        feature_groups=data.feature_groups,
        column_names=data.column_names,
    )

    enc_trainer = Trainer(fast_dev_run=True, logger=False)
    enc_trainer.fit(encoder, datamodule=data)
    enc_trainer.test(ckpt_path=None, datamodule=data)

    classifier = Clf(
        CLF,
        num_s=data.num_s,
        data_dim=data.data_dim,
        s_dim=data.s_dim,
        cf_available=data.cf_available,
        outcome_cols=data.outcome_columns,
    )

    clf_trainer = Trainer(fast_dev_run=True, logger=False)
    clf_trainer.fit(classifier, datamodule=data)
    clf_trainer.test(ckpt_path=None, datamodule=data)

    model = AiesModel(encoder=encoder, classifier=classifier)
    model_trainer = Trainer(fast_dev_run=True, logger=False)
    model_trainer.fit(model, datamodule=data)
    model_trainer.test(ckpt_path=None, datamodule=data)