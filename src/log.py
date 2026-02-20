# log.py — Centralized logging for CuaOS
from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler

_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(_LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(_LOG_DIR, "cuaos.log")

_FORMAT = "[%(asctime)s] %(levelname)-7s %(name)-14s  %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def _setup_root() -> None:
    """Configure the root 'cuaos' logger once."""
    root = logging.getLogger("cuaos")
    if root.handlers:
        return  # already configured

    root.setLevel(logging.DEBUG)

    # File handler — rotate at 5 MB, keep 3 backups
    fh = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3,
                             encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATE_FMT))
    root.addHandler(fh)

    # Console handler — INFO and above
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATE_FMT))
    root.addHandler(ch)


_setup_root()


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the 'cuaos' namespace."""
    return logging.getLogger(f"cuaos.{name}")
