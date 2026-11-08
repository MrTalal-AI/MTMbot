"""Logging setup shared by the CLI commands."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

__all__ = ["setup_logging"]

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_DATEFMT = "%Y-%m-%dT%H:%M:%SZ"


def setup_logging(level: str = "INFO", log_file: str | Path | None = None) -> None:
    """Configure root logging once, to stderr and optionally a file."""
    numeric = getattr(logging, level.upper(), None)
    if not isinstance(numeric, int):
        raise ValueError(f"invalid log level {level!r}")
    handlers: list[logging.Handler] = [logging.StreamHandler(stream=sys.stderr)]
    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, encoding="utf-8"))
    logging.basicConfig(
        level=numeric,
        format=_FORMAT,
        datefmt=_DATEFMT,
        handlers=handlers,
        force=True,
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)
