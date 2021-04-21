"""Logging functions."""
import logging
from typing import Any, Optional

from pytorch_lightning.loggers import WandbLogger

log = logging.getLogger(__name__)


def do_log(name: str, val: Any, logger: Optional[WandbLogger]) -> None:
    """Log to experiment tracker and also the logger."""
    if isinstance(val, (float, int)):
        log.info(f"{name}: {val}")
    if logger is not None:
        logger.experiment.log({name: val})
