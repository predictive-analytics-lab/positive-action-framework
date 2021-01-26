"""Scoring functions."""
from copy import copy

import numpy as np
import pandas as pd
from ethicml import Accuracy, DataTuple, Prediction
from pytorch_lightning.loggers import LightningLoggerBase
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import KFold

from src.data_modules.base_module import BaseDataModule
from src.logging import do_log
from src.model.common_model import CommonModel
from src.model.encoder_model import AE


def lrcv_results(
    train: np.ndarray,
    test: np.ndarray,
    dm: BaseDataModule,
    logger: LightningLoggerBase,
    component: str,
) -> None:
    """Run an LRCV over some train set and apply to some test set."""
    for train_target, test_target, target_name in [
        (dm.train_data.s, dm.test_data.s, "S"),
        (dm.train_data.y, dm.test_data.y, "Y"),
    ]:
        random_state = np.random.RandomState(888)
        folder = KFold(n_splits=5, shuffle=True, random_state=random_state)
        clf = LogisticRegressionCV(
            cv=folder,
            n_jobs=-1,
            random_state=random_state,
            solver="liblinear",
            multi_class="auto",
        )
        clf.fit(train, train_target.to_numpy().ravel())
        s_preds = Prediction(hard=pd.Series(clf.predict(test)), info=dict(C=clf.C_[0]))
        do_log(
            f"Baselines/LRCV/Accuracy-{target_name}-from-{component}",
            Accuracy().score(prediction=s_preds, actual=dm.test_data.replace(y=test_target)),
            logger,
        )


def produce_baselines(*, encoder: CommonModel, dm: BaseDataModule, logger: LightningLoggerBase) -> None:
    """Produce baselines for predictiveness."""
    latent_train = encoder.get_latent(dm.train_dataloader(shuffle=False, drop_last=False))
    latent_test = encoder.get_latent(dm.test_dataloader())
    lrcv_results(latent_train, latent_test, dm, logger, f"{encoder.model_name}-Z")

    if isinstance(encoder, AE):
        train = dm.train_data.x.to_numpy()
        test = dm.test_data.x.to_numpy()
        lrcv_results(train, test, dm, logger, "Og-Data")
        recon_name = "Recon-Data"
    else:
        train_labels = dm.train_data.y.to_numpy()
        test_labels = dm.test_data.y.to_numpy()
        lrcv_results(train_labels, test_labels, dm, logger, "Og-Labels")
        recon_name = "Preds"

    train_recon = encoder.get_recon(dm.train_dataloader(shuffle=False, drop_last=False))
    test_recon = encoder.get_recon(dm.test_dataloader())
    lrcv_results(train_recon, test_recon, dm, logger, recon_name)


def get_miri_metrics(
    method: str,
    acceptance: DataTuple,
    graduated: DataTuple,
    logger,
    y_denotation: str = "Y",
    s_denotation: str = "S",
    ty_denotation: str = "Ty",
) -> None:
    """Get miri requested metrics."""
    data = copy(acceptance)
    data_y_true = copy(graduated)

    num_points = data.y.shape[0]
    sum_y_is_ty = sum((data.y.values - data_y_true.y.values) == 0)[0]

    do_log(f"{method}/P({y_denotation}={ty_denotation})", sum_y_is_ty / num_points, logger)

    sum_y_is_ty_given_s0 = sum(
        (data.y[data.s[data.s.columns[0]] == 0].values - data_y_true.y[data.s[data.s.columns[0]] == 0].values) == 0
    )[0]
    num_s0 = data.y[data.s[data.s.columns[0]] == 0].shape[0]

    sum_y_is_ty_given_s1 = sum(
        (data.y[data.s[data.s.columns[0]] == 1].values - data_y_true.y[data.s[data.s.columns[0]] == 1].values) == 0
    )[0]
    num_s1 = data.y[data.s[data.s.columns[0]] == 1].shape[0]

    do_log(f"{method}/P({y_denotation}={ty_denotation}|{s_denotation}=0)", sum_y_is_ty_given_s0 / num_s0, logger)
    do_log(f"{method}/P({y_denotation}={ty_denotation}|{s_denotation}=1)", sum_y_is_ty_given_s1 / num_s1, logger)

    for val in [0, 1]:
        # P(y^=val)
        result = data.y[data.y.columns[0]] == val
        do_log(f"{method}/P({y_denotation}={val})", result.sum() / result.count(), logger)

        # P(Ty=val)
        result = data_y_true.y[data_y_true.y.columns[0]] == val
        do_log(f"{method}/P({ty_denotation}={val})", result.sum() / result.count(), logger)

    for outer_val in [0, 1]:
        for inner_val in [0, 1]:
            # P(y^=outer | S=inner)
            result = data.y[data.s[data.s.columns[0]] == inner_val][data.y.columns[0]] == outer_val
            do_log(
                f"{method}/P({y_denotation}={outer_val}|{s_denotation}={inner_val})",
                result.sum() / result.count(),
                logger,
            )

            # P(y^=outer | Ty=inner)
            result = data.y[data_y_true.y[data_y_true.y.columns[0]] == inner_val][data.y.columns[0]] == outer_val
            do_log(
                f"{method}/P({y_denotation}={outer_val}|{ty_denotation}={inner_val})",
                result.sum() / result.count(),
                logger,
            )

            # P(Ty=outer | y^=inner)
            result = data_y_true.y[data.y[data.y.columns[0]] == outer_val][data_y_true.y.columns[0]] == inner_val
            do_log(
                f"{method}/P({ty_denotation}={inner_val}|{y_denotation}={outer_val})",
                result.sum() / result.count(),
                logger,
            )

            result = data_y_true.y[data.s[data.s.columns[0]] == outer_val][data_y_true.y.columns[0]] == inner_val
            do_log(
                f"{method}/P({ty_denotation}={inner_val}|{s_denotation}={outer_val})",
                result.sum() / result.count(),
                logger,
            )

            result = data.y[data.s[data.s.columns[0]] == outer_val][data.y.columns[0]] == inner_val
            do_log(
                f"{method}/P({y_denotation}={inner_val}|{s_denotation}={outer_val})",
                result.sum() / result.count(),
                logger,
            )

            result = data_y_true.y[data.s[data.s.columns[0]] == outer_val][data_y_true.y.columns[0]] == inner_val
            do_log(
                f"{method}/P({ty_denotation}={inner_val}|{s_denotation}={outer_val})",
                result.sum() / result.count(),
                logger,
            )

    for s_val in [0, 1]:
        for ty_val in [0, 1]:
            for y_val in [0, 1]:
                result = (
                    data_y_true.y[(data.y[data.y.columns[0]] == y_val) & (data.s[data.s.columns[0]] == s_val)][
                        data_y_true.y.columns[0]
                    ]
                    == ty_val
                )
                do_log(
                    f"{method}/P({ty_denotation}={ty_val}|{s_denotation}={s_val},{y_denotation}={y_val})",
                    result.sum() / result.count(),
                    logger,
                )
