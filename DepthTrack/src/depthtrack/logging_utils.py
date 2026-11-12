"""Centralised logging configuration.

Replaces ad-hoc ``print`` calls with the standard ``logging`` module so output
can be filtered by level, redirected, or silenced by downstream consumers.
"""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False

_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


def configure_logging(level: str = "INFO") -> None:
    """Configure the root logger once with a concise, readable format."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    log_level = _LEVELS.get(level.upper(), logging.INFO)
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root = logging.getLogger()
    root.setLevel(log_level)
    root.handlers.clear()
    root.addHandler(handler)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger (configures logging lazily if needed)."""
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(name)
