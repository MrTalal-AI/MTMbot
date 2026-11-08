"""Kraken public OHLC feed (no API key required).

Kraken returns at most 720 bars per request, which is plenty for the live
engine (warm-up is a few hundred bars) but not for years of history; use the
Coinbase source or ``mtmbot download`` for that.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from mtmbot.data.base import DataError, sanitize
from mtmbot.data.http import get_json
from mtmbot.models import Candle

__all__ = ["KrakenSource"]

_BASE_URL = "https://api.kraken.com/0/public/OHLC"
_ALLOWED_STEPS = {60: 1, 300: 5, 900: 15, 3600: 60, 14400: 240, 86400: 1440}
_PAIR_ALIASES = {"BTC": "XBT", "DOGE": "XDG"}


def _kraken_pair(symbol: str) -> str:
    if "/" not in symbol:
        raise DataError(f"symbol must look like BASE/QUOTE, got {symbol!r}")
    base, quote = symbol.split("/", 1)
    base = _PAIR_ALIASES.get(base.upper(), base.upper())
    return f"{base}{quote.upper()}"


class KrakenSource:
    """Recent closed candles from Kraken's public API."""

    name = "kraken"

    def fetch_candles(self, symbol: str, step_seconds: int, limit: int = 500) -> list[Candle]:
        """Return up to ``limit`` recent closed candles, oldest first."""
        if step_seconds not in _ALLOWED_STEPS:
            supported = ", ".join(
                f"{seconds}s" for seconds in sorted(_ALLOWED_STEPS)
            )
            raise DataError(
                f"kraken supports only these timeframes: {supported}; got {step_seconds}s"
            )
        if limit < 1:
            raise DataError(f"limit must be >= 1, got {limit!r}")

        payload: Any = get_json(
            _BASE_URL,
            {"pair": _kraken_pair(symbol), "interval": _ALLOWED_STEPS[step_seconds]},
        )
        if not isinstance(payload, dict):
            raise DataError(f"kraken returned an unexpected payload for {symbol}")
        errors = payload.get("error") or []
        if errors:
            raise DataError(f"kraken error for {symbol}: {', '.join(str(e) for e in errors)}")
        result = payload.get("result")
        if not isinstance(result, dict):
            raise DataError(f"kraken returned no result for {symbol}")

        rows: list[list[Any]] | None = None
        for key, value in result.items():
            if key == "last":
                continue
            if isinstance(value, list):
                rows = value
                break
        if not rows:
            raise DataError(f"kraken returned no candles for {symbol}")

        candles: list[Candle] = []
        for row in rows:
            if not isinstance(row, list) or len(row) < 6:
                raise DataError(f"kraken row is malformed for {symbol}: {row!r}")
            candles.append(
                Candle(
                    ts=datetime.fromtimestamp(int(row[0]), tz=timezone.utc),
                    open=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                    volume=float(row[6]) if len(row) > 6 else float(row[5]),
                )
            )

        cleaned = sanitize(candles, step_seconds)
        if not cleaned:
            raise DataError(f"kraken returned no closed candles for {symbol}")
        return cleaned[-limit:]
