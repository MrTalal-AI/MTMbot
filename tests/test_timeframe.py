from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from mtmbot.timeframe import (
    TimeframeError,
    bars_per_year,
    buckets_between,
    ceil_ts,
    floor_ts,
    format_timeframe,
    next_bar_close,
    parse_timeframe,
)


@pytest.mark.parametrize(
    ("text", "seconds"),
    [("1m", 60), ("5m", 300), ("15m", 900), ("1h", 3600), ("4h", 14400), ("1d", 86400)],
)
def test_parse_timeframe_valid(text: str, seconds: int) -> None:
    assert parse_timeframe(text) == seconds


@pytest.mark.parametrize("text", ["", "0m", "15", "m15", "-5m", "1.5h", "1y", "  "])
def test_parse_timeframe_invalid(text: str) -> None:
    with pytest.raises(TimeframeError):
        parse_timeframe(text)


def test_parse_timeframe_rejects_non_string() -> None:
    with pytest.raises(TimeframeError):
        parse_timeframe(60)  # type: ignore[arg-type]


@pytest.mark.parametrize("seconds", [60, 300, 3600, 86400, 604800])
def test_format_roundtrip(seconds: int) -> None:
    assert parse_timeframe(format_timeframe(seconds)) == seconds


def test_format_rejects_non_divisible() -> None:
    with pytest.raises(TimeframeError):
        format_timeframe(0)


def test_floor_and_ceil() -> None:
    moment = datetime(2024, 5, 1, 10, 37, 42, tzinfo=UTC)
    floored = floor_ts(moment, 3600)
    assert floored == datetime(2024, 5, 1, 10, 0, tzinfo=UTC)
    assert ceil_ts(moment, 3600) == datetime(2024, 5, 1, 11, 0, tzinfo=UTC)
    assert ceil_ts(floored, 3600) == floored


def test_next_bar_close_adds_grace() -> None:
    moment = datetime(2024, 5, 1, 10, 0, tzinfo=UTC)
    assert next_bar_close(moment, 3600, grace_seconds=2) == (
        datetime(2024, 5, 1, 11, 0, tzinfo=UTC) + timedelta(seconds=2)
    )


def test_bars_per_year() -> None:
    assert bars_per_year(3600) == pytest.approx(8760.0)


def test_buckets_between() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    assert buckets_between(start, start + timedelta(hours=5), 3600) == 5
    assert buckets_between(start, start, 3600) == 0


def test_floor_rejects_bad_step() -> None:
    with pytest.raises(TimeframeError):
        floor_ts(datetime.now(UTC), 0)
