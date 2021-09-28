"""Logging functions."""
from __future__ import annotations
import logging
from typing import Any

import pytorch_lightning.loggers as pll

log = logging.getLogger(__name__)


def do_log(name: str, val: Any, logger: pll.WandbLogger | pll.base.DummyLogger) -> None:
    """Log to experiment tracker and also the logger."""
    if isinstance(val, (float, int)):
        log.info(f"{name}: {val}")
        # print(f"{name}: {val}")
    if isinstance(logger, pll.WandbLogger):
        logger.experiment.summary[f"{name}"] = val
