"""Logging functions."""
import logging
from typing import Any, Optional, Union

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers.base import DummyLogger

log = logging.getLogger(__name__)


def do_log(name: str, val: Any, logger: Optional[Union[WandbLogger, DummyLogger]]) -> None:
    """Log to experiment tracker and also the logger."""
    if isinstance(val, (float, int)):
        log.info(f"{name}: {val}")
    if logger is not None and not isinstance(logger, DummyLogger):
        logger.experiment.log({name: val})
