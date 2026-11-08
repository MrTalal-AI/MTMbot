"""Coinbase public candles: history download and recent-candle polling.

The public endpoint needs no credentials and supports pagination, which makes
it the default source for backtests.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Any

from mtmbot.data.base import DataError, sanitize
from mtmbot.data.http import get_json
from mtmbot.data.resample import resample
from mtmbot.models import Candle

__all__ = ["CoinbaseSource", "download_history"]

_BASE_URL = "https://api.exchange.coinbase.com"
_GRANULARITIES = {60, 300, 900, 3600, 21600, 86400}
_MAX_BARS_PER_REQUEST = 300
_MIN_INTERVAL = 0.34  # seconds between requests, well inside the public rate limit


def _coinbase_product(symbol: str) -> str:
    if "/" not in symbol:
        raise DataError(f"symbol must look like BASE/QUOTE, got {symbol!r}")
    base, quote = symbol.split("/", 1)
    return f"{base.upper()}-{quote.upper()}"


def _granularity_for(step_seconds: int) -> int:
    """Best Coinbase granularity for a requested bar size (finer if needed)."""
    for granularity in sorted(_GRANULARITIES, reverse=True):
        if step_seconds % granularity == 0:
            return granularity
    supported = ", ".join(f"{seconds}s" for seconds in sorted(_GRANULARITIES))
    raise DataError(
        "coinbase supports timeframes that are multiples of one of: "
        f"{supported}; got {step_seconds}s"
    )


def _row_to_candle(row: list[Any], product: str) -> Candle:
    # Coinbase returns [time, low, high, open, close, volume], newest first.
    if not isinstance(row, list) or len(row) < 6:
        raise DataError(f"coinbase row is malformed for {product}: {row!r}")
    return Candle(
        ts=datetime.fromtimestamp(int(row[0]), tz=timezone.utc),
        open=float(row[3]),
        high=float(row[2]),
        low=float(row[1]),
        close=float(row[4]),
        volume=float(row[5]),
    )


class CoinbaseSource:
    """Coinbase Exchange public market data."""

    name = "coinbase"

    def fetch_range(
        self, symbol: str, step_seconds: int, start: datetime, end: datetime
    ) -> list[Candle]:
        """Return all closed candles inside ``[start, end]``, paginated."""
        granularity = _granularity_for(step_seconds)
        factor = step_seconds // granularity
        if start.tzinfo is None or end.tzinfo is None:
            raise DataError("start and end must be timezone-aware")
        start = start.astimezone(timezone.utc)
        end = end.astimezone(timezone.utc)
        if end <= start:
            raise DataError("end must be after start")

        product = _coinbase_product(symbol)
        url = f"{_BASE_URL}/products/{product}/candles"
        window = timedelta(seconds=granularity * _MAX_BARS_PER_REQUEST)
        collected: list[Candle] = []
        cursor = start
        while cursor < end:
            chunk_end = min(cursor + window, end)
            payload: Any = get_json(
                url,
                {
                    "granularity": granularity,
                    "start": cursor.isoformat().replace("+00:00", "Z"),
                    "end": chunk_end.isoformat().replace("+00:00", "Z"),
                },
            )
            if not isinstance(payload, list):
                raise DataError(f"coinbase returned an unexpected payload for {product}")
            for row in payload:
                collected.append(_row_to_candle(row, product))
            cursor = chunk_end
            time.sleep(_MIN_INTERVAL)

        cleaned = sanitize(collected, granularity)
        if not cleaned:
            raise DataError(f"coinbase returned no candles for {product} in the requested range")
        if factor > 1:
            cleaned = resample(cleaned, granularity, factor)
            if not cleaned:
                raise DataError(
                    f"not enough {granularity}s history to build {step_seconds}s bars for {product}"
                )
        return cleaned

    def fetch_candles(self, symbol: str, step_seconds: int, limit: int = 500) -> list[Candle]:
        """Return up to ``limit`` recent closed candles, oldest first."""
        if limit < 1:
            raise DataError(f"limit must be >= 1, got {limit!r}")
        now = datetime.now(timezone.utc)
        span = timedelta(seconds=step_seconds * min(limit + 30, _MAX_BARS_PER_REQUEST))
        candles = self.fetch_range(symbol, step_seconds, now - span, now)
        return candles[-limit:]


def download_history(
    symbol: str,
    step_seconds: int,
    start: datetime,
    end: datetime,
    *,
    source: str = "coinbase",
) -> list[Candle]:
    """Download history from the named source (currently only Coinbase)."""
    if source != "coinbase":
        raise DataError(
            f"history download supports source 'coinbase' only, got {source!r} "
            "(kraken only serves the latest 720 bars)"
        )
    candles = CoinbaseSource().fetch_range(symbol, step_seconds, start, end)
    return candles
