"""Scoring functions."""
from __future__ import annotations
from copy import copy
import logging

from ethicml import Accuracy, DataTuple, Prediction
import numpy as np
import numpy.typing as npt
import pandas as pd
import pytorch_lightning.loggers as pll
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import KFold

from paf.architectures.model.model_components import AE, CommonModel
from paf.base_templates.base_module import BaseDataModule
from paf.log_progress import do_log


def lrcv_results(
    *,
    train: npt.NDArray[np.float32],
    test: npt.NDArray[np.float32],
    datamodule: BaseDataModule,
    logger: pll.LightningLoggerBase,
    component: str,
    test_mode: bool,
) -> None:
    """Run an LRCV over some train set and apply to some test set."""
    for train_target, test_target, target_name in [
        (datamodule.train_datatuple.s, datamodule.test_datatuple.s, "S"),
        (datamodule.train_datatuple.y, datamodule.test_datatuple.y, "Y"),
    ]:
        random_state = np.random.RandomState(888)
        folder = KFold(n_splits=5, shuffle=True, random_state=random_state)
        clf = LogisticRegressionCV(
            cv=folder,
            n_jobs=-1,
            random_state=random_state,
            solver="liblinear",
            multi_class="auto",
            max_iter=1 if test_mode else 100,
            tol=1 if test_mode else 1e-4,
        )
        clf.fit(train, train_target.to_numpy().ravel())
        s_preds = Prediction(hard=pd.Series(clf.predict(test)), info=dict(C=clf.C_[0]))
        do_log(
            f"Baselines/LRCV/Accuracy-{target_name}-from-{component}",
            Accuracy().score(
                prediction=s_preds, actual=datamodule.test_datatuple.replace(y=test_target)
            ),
            logger,
        )


def produce_baselines(
    *,
    encoder: CommonModel,
    datamodule: BaseDataModule,
    logger: pll.LightningLoggerBase,
    test_mode: bool,
) -> None:
    """Produce baselines for predictiveness."""
    latent_train = encoder.get_latent(datamodule.train_dataloader(shuffle=False, drop_last=False))
    latent_test = encoder.get_latent(datamodule.test_dataloader())
    lrcv_results(
        train=latent_train,
        test=latent_test,
        datamodule=datamodule,
        logger=logger,
        component=f"{encoder.model_name}-Z",
        test_mode=test_mode,
    )

    if isinstance(encoder, AE):
        train = datamodule.train_datatuple.x.to_numpy()
        test = datamodule.test_datatuple.x.to_numpy()
        lrcv_results(
            train=train,
            test=test,
            datamodule=datamodule,
            logger=logger,
            component="Og-Data",
            test_mode=test_mode,
        )
        recon_name = "Recon-Data"
    else:
        train_labels = datamodule.train_datatuple.y.to_numpy()
        test_labels = datamodule.test_datatuple.y.to_numpy()
        lrcv_results(
            train=train_labels,
            test=test_labels,
            datamodule=datamodule,
            logger=logger,
            component="Og-Labels",
            test_mode=test_mode,
        )
        recon_name = "Preds"

    train_recon = encoder.get_recon(datamodule.train_dataloader(shuffle=False, drop_last=False))
    test_recon = encoder.get_recon(datamodule.test_dataloader())
    lrcv_results(
        train=train_recon,
        test=test_recon,
        datamodule=datamodule,
        logger=logger,
        component=recon_name,
        test_mode=test_mode,
    )


def get_full_breakdown(
    method: str,
    acceptance: DataTuple,
    graduated: DataTuple,
    logger: logging.Logger | None,
    y_denotation: str = "Y",
    s_denotation: str = "S",
    ty_denotation: str = "Ty",
) -> None:
    """Get full array of statistics."""
    data = copy(acceptance)
    data_y_true = copy(graduated)

    num_points = data.y.shape[0]
    sum_y_is_ty = sum((data.y.values - data_y_true.y.values) == 0)[0]  # type: ignore[index]

    do_log(f"{method}/P({y_denotation}={ty_denotation})", sum_y_is_ty / num_points, logger)

    sum_y_is_ty_given_s0 = sum(  # type: ignore[index]
        (
            data.y[data.s[data.s.columns[0]] == 0].values
            - data_y_true.y[data.s[data.s.columns[0]] == 0].values
        )
        == 0
    )[0]
    num_s0 = data.y[data.s[data.s.columns[0]] == 0].shape[0]

    sum_y_is_ty_given_s1 = sum(  # type: ignore[index]
        (
            data.y[data.s[data.s.columns[0]] == 1].values
            - data_y_true.y[data.s[data.s.columns[0]] == 1].values
        )
        == 0
    )[0]
    num_s1 = data.y[data.s[data.s.columns[0]] == 1].shape[0]

    do_log(
        f"{method}/P({y_denotation}={ty_denotation}|{s_denotation}=0)",
        sum_y_is_ty_given_s0 / num_s0,
        logger,
    )
    do_log(
        f"{method}/P({y_denotation}={ty_denotation}|{s_denotation}=1)",
        sum_y_is_ty_given_s1 / num_s1,
        logger,
    )

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
            result = (
                data.y[data_y_true.y[data_y_true.y.columns[0]] == inner_val][data.y.columns[0]]
                == outer_val
            )
            do_log(
                f"{method}/P({y_denotation}={outer_val}|{ty_denotation}={inner_val})",
                result.sum() / result.count(),
                logger,
            )

            # P(Ty=outer | y^=inner)
            result = (
                data_y_true.y[data.y[data.y.columns[0]] == outer_val][data_y_true.y.columns[0]]
                == inner_val
            )
            do_log(
                f"{method}/P({ty_denotation}={inner_val}|{y_denotation}={outer_val})",
                result.sum() / result.count(),
                logger,
            )

            result = (
                data_y_true.y[data.s[data.s.columns[0]] == outer_val][data_y_true.y.columns[0]]
                == inner_val
            )
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

            result = (
                data_y_true.y[data.s[data.s.columns[0]] == outer_val][data_y_true.y.columns[0]]
                == inner_val
            )
            do_log(
                f"{method}/P({ty_denotation}={inner_val}|{s_denotation}={outer_val})",
                result.sum() / result.count(),
                logger,
            )

    for s_val in [0, 1]:
        for ty_val in [0, 1]:
            for y_val in [0, 1]:
                result = (
                    data_y_true.y[
                        (data.y[data.y.columns[0]] == y_val) & (data.s[data.s.columns[0]] == s_val)
                    ][data_y_true.y.columns[0]]
                    == ty_val
                )
                do_log(
                    f"{method}/P({ty_denotation}={ty_val}|{s_denotation}={s_val},{y_denotation}={y_val})",
                    result.sum() / result.count(),
                    logger,
                )

                result = (
                    data.y[
                        (data_y_true.y[data_y_true.y.columns[0]] == ty_val)
                        & (data.s[data.s.columns[0]] == s_val)
                    ][data.y.columns[0]]
                    == y_val
                )
                do_log(
                    f"{method}/P({y_denotation}={y_val}|{s_denotation}={s_val},{ty_denotation}={ty_val})",
                    result.sum() / result.count(),
                    logger,
                )
