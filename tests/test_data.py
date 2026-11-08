from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from mtmbot.data import (
    CoinbaseSource,
    DataError,
    KrakenSource,
    map_htf_indices,
    read_candles,
    resample,
    sanitize,
    write_candles,
)
from mtmbot.models import Candle

START = datetime(2024, 1, 1, tzinfo=UTC)


def _candle(hour: int, o: float, h: float, low: float, c: float, v: float = 1.0) -> Candle:
    return Candle(ts=START + timedelta(hours=hour), open=o, high=h, low=low, close=c, volume=v)


def test_resample_aggregates_ohlcv() -> None:
    candles = [
        _candle(0, 100, 102, 99, 101, 1),
        _candle(1, 101, 105, 100, 104, 2),
        _candle(2, 104, 106, 103, 105, 3),
        _candle(3, 105, 107, 104, 106, 4),
    ]
    bars = resample(candles, 3600, 2)
    assert len(bars) == 2
    assert bars[0].open == 100
    assert bars[0].close == 104
    assert bars[0].high == 105
    assert bars[0].low == 99
    assert bars[0].volume == 3
    assert bars[1].open == 104
    assert bars[1].close == 106
    assert bars[1].volume == 7


def test_resample_drops_incomplete_bucket() -> None:
    candles = [_candle(0, 100, 101, 99, 100.5), _candle(1, 100.5, 101, 99, 100.8),
               _candle(2, 100.8, 102, 100, 101.5)]
    assert len(resample(candles, 3600, 2)) == 1


def test_resample_rejects_bad_factor() -> None:
    with pytest.raises(ValueError):
        resample([_candle(0, 1, 1, 1, 1)], 3600, 1)


def test_map_htf_indices_never_sees_open_bars() -> None:
    base = [_candle(hour, 100, 101, 99, 100) for hour in range(6)]
    htf = [_candle(0, 100, 102, 99, 101), _candle(2, 101, 103, 100, 102)]
    mapping = map_htf_indices(base, htf, 7200)
    assert mapping == [-1, -1, 0, 0, 1, 1]


def test_sanitize_sorts_and_dedupes() -> None:
    early = _candle(0, 100, 101, 99, 100)
    duplicate = Candle(ts=early.ts, open=100, high=103, low=98, close=102)
    later = _candle(1, 102, 104, 101, 103)
    cleaned = sanitize([early, duplicate, later], 3600)
    assert [c.ts for c in cleaned] == [early.ts, later.ts]
    assert cleaned[0].close == 102


def test_sanitize_drops_forming_bar() -> None:
    now = START + timedelta(minutes=30)
    forming = _candle(0, 100, 101, 99, 100)
    assert sanitize([forming], 3600, now) == []


def test_csv_roundtrip(tmp_path) -> None:
    candles = [_candle(hour, 100 + hour, 101 + hour, 99 + hour, 100.5 + hour) for hour in range(4)]
    path = write_candles(tmp_path / "data.csv", candles)
    loaded = read_candles(path)
    assert loaded == candles


def test_csv_missing_file() -> None:
    with pytest.raises(DataError, match="not found"):
        read_candles("/nope/missing.csv")


def test_csv_rejects_bad_value(tmp_path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text(
        "ts,open,high,low,close,volume\n2024-01-01T00:00:00+00:00,abc,1,1,1,1\n",
        encoding="utf-8",
    )
    with pytest.raises(DataError, match="not a number"):
        read_candles(path)


def test_csv_rejects_missing_column(tmp_path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("ts,open,high\n2024-01-01T00:00:00+00:00,1,1\n", encoding="utf-8")
    with pytest.raises(DataError, match="missing column"):
        read_candles(path)


def test_empty_candles_not_written(tmp_path) -> None:
    with pytest.raises(DataError):
        write_candles(tmp_path / "empty.csv", [])


def test_sources_reject_unsupported_timeframes_without_network() -> None:
    with pytest.raises(DataError, match="kraken supports"):
        KrakenSource().fetch_candles("BTC/USD", 7)
    with pytest.raises(DataError, match="coinbase supports"):
        CoinbaseSource().fetch_range("BTC/USD", 7, START, START + timedelta(days=1))
