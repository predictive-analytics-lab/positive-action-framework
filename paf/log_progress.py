"""Logging functions."""
from __future__ import annotations
import logging
from typing import Any

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers.base import DummyLogger

log = logging.getLogger(__name__)


def do_log(name: str, val: Any, logger: WandbLogger | DummyLogger) -> None:
    """Log to experiment tracker and also the logger."""
    if isinstance(val, (float, int)):
        log.info(f"{name}: {val}")
        # print(f"{name}: {val}")
    if isinstance(logger, WandbLogger):
        logger.experiment.log({name: val})
