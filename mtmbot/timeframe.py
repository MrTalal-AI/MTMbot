"""Timeframe parsing and bar-alignment helpers.

All timestamps inside MTMbot are timezone-aware UTC datetimes. Bars are
identified by their *open* time, which is always aligned to the start of the
timeframe bucket (e.g. an hourly bar starts at :00:00).
"""

from __future__ import annotations

import math
import re
from datetime import UTC, datetime, timedelta

_UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}
_TIMEFRAME_RE = re.compile(r"^(?P<value>[1-9][0-9]*)(?P<unit>[smhdw])$")


class TimeframeError(ValueError):
    """Raised when a timeframe string cannot be parsed."""


def parse_timeframe(text: str) -> int:
    """Parse a timeframe such as ``15m``, ``1h`` or ``4h`` into seconds."""
    if not isinstance(text, str):
        raise TimeframeError(f"timeframe must be a string, got {type(text).__name__}")
    match = _TIMEFRAME_RE.match(text.strip().lower())
    if match is None:
        raise TimeframeError(
            f"invalid timeframe {text!r}; expected a number followed by s/m/h/d/w, e.g. '15m'"
        )
    seconds = int(match.group("value")) * _UNIT_SECONDS[match.group("unit")]
    if seconds < 1:
        raise TimeframeError(f"timeframe {text!r} is too small")
    return seconds


def format_timeframe(seconds: int) -> str:
    """Render a timeframe in seconds back to its canonical short form."""
    if seconds <= 0 or seconds % 1:
        raise TimeframeError(f"timeframe must be a positive integer of seconds, got {seconds!r}")
    for unit in ("w", "d", "h", "m", "s"):
        size = _UNIT_SECONDS[unit]
        if seconds % size == 0:
            return f"{seconds // size}{unit}"
    raise TimeframeError(f"cannot format timeframe {seconds!r}")


def floor_ts(ts: datetime, seconds: int) -> datetime:
    """Floor a timestamp down to the start of its ``seconds`` bucket (UTC)."""
    if seconds <= 0:
        raise TimeframeError(f"step must be positive, got {seconds!r}")
    epoch = int(ts.timestamp())
    return datetime.fromtimestamp(epoch - epoch % seconds, tz=UTC)


def ceil_ts(ts: datetime, seconds: int) -> datetime:
    """Round a timestamp up to the next bucket boundary (UTC)."""
    floored = floor_ts(ts, seconds)
    return floored if floored == ts else floored + timedelta(seconds=seconds)


def next_bar_close(now: datetime, seconds: int, grace_seconds: float = 2.0) -> datetime:
    """Return when the current bar closes, plus a small grace period.

    Used by the live engine so it polls right after a bar closes instead of
    spinning in a hot loop.
    """
    if seconds <= 0:
        raise TimeframeError(f"step must be positive, got {seconds!r}")
    if grace_seconds < 0:
        raise ValueError(f"grace_seconds must be >= 0, got {grace_seconds!r}")
    # Strictly the *next* boundary: if we are exactly at one, the bar that just
    # opened needs a full step to close.
    return floor_ts(now, seconds) + timedelta(seconds=seconds + grace_seconds)


def bars_per_year(seconds: int) -> float:
    """Number of bars of this timeframe in a (365 day) year."""
    if seconds <= 0:
        raise TimeframeError(f"step must be positive, got {seconds!r}")
    return (365 * 24 * 3600) / seconds


def buckets_between(start: datetime, end: datetime, seconds: int) -> int:
    """How many whole bars fit between two timestamps."""
    if seconds <= 0:
        raise TimeframeError(f"step must be positive, got {seconds!r}")
    span = (end - start).total_seconds()
    if span <= 0:
        return 0
    return math.floor(span / seconds)
