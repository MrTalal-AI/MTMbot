"""Market data access.

All sources return ``Candle`` objects in UTC, sorted by open time, with the
still-forming bar removed. That contract is what lets the strategy treat the
last candle as "closed and final".
"""

from __future__ import annotations

from mtmbot.data.base import (
    CandleSource,
    DataError,
    HistoricalSource,
    sanitize,
)
from mtmbot.data.coinbase import CoinbaseSource, download_history
from mtmbot.data.csv_io import read_candles, write_candles
from mtmbot.data.kraken import KrakenSource
from mtmbot.data.resample import map_htf_indices, resample

__all__ = [
    "CandleSource",
    "CoinbaseSource",
    "DataError",
    "HistoricalSource",
    "KrakenSource",
    "download_history",
    "map_htf_indices",
    "read_candles",
    "resample",
    "sanitize",
    "write_candles",
]
