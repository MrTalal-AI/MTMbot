"""Data-source contracts and integrity checks."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

from mtmbot.models import Candle

__all__ = ["CandleSource", "DataError", "HistoricalSource", "sanitize"]


class DataError(RuntimeError):
    """Raised when market data is unavailable or fails validation."""


@runtime_checkable
class CandleSource(Protocol):
    """Something that can return recent closed candles."""

    name: str

    def fetch_candles(self, symbol: str, step_seconds: int, limit: int = 500) -> list[Candle]:
        """Return up to ``limit`` most recent *closed* candles, oldest first."""
        ...


@runtime_checkable
class HistoricalSource(Protocol):
    """Something that can return a date range of candles."""

    name: str

    def fetch_range(
        self, symbol: str, step_seconds: int, start: datetime, end: datetime
    ) -> list[Candle]:
        """Return all closed candles fully inside ``[start, end]``."""
        ...


def sanitize(candles: list[Candle], step_seconds: int, now: datetime | None = None) -> list[Candle]:
    """Sort, de-duplicate and drop the still-forming bar.

    A duplicate open time means the source re-sent a bar (common when paging an
    API); the later copy wins, which matches exchange behaviour. Bars whose
    whole interval has not elapsed yet are dropped so the strategy never reads
    a half-finished candle.
    """
    if step_seconds <= 0:
        raise DataError(f"step_seconds must be positive, got {step_seconds!r}")
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise DataError("now must be timezone-aware")
    current = current.astimezone(timezone.utc)

    by_ts: dict[datetime, Candle] = {}
    for candle in candles:
        by_ts[candle.ts] = candle

    ordered = [by_ts[key] for key in sorted(by_ts)]
    closed = [
        candle
        for candle in ordered
        if candle.ts.timestamp() + step_seconds <= current.timestamp() + 1e-6
    ]
    return closed
