from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from mtmbot.models import Candle


def make_candles(
    closes: list[float],
    *,
    step: int = 3600,
    start: datetime = datetime(2024, 1, 1, tzinfo=UTC),
    wick: float = 0.0005,
) -> list[Candle]:
    """Build a valid candle series where each bar opens at the previous close."""
    candles: list[Candle] = []
    previous = closes[0]
    for index, close in enumerate(closes):
        high = max(previous, close) * (1.0 + wick)
        low = min(previous, close) * (1.0 - wick)
        candles.append(
            Candle(
                ts=start + timedelta(seconds=step * index),
                open=previous,
                high=high,
                low=low,
                close=close,
                volume=1000.0 + index,
            )
        )
        previous = close
    return candles


def trending_closes(length: int = 700, drift: float = 0.0025, pullback: tuple[int, int] = (300, 310)) -> list[float]:
    """A long uptrend with one shallow pullback, to force momentum re-entries."""
    closes: list[float] = []
    price = 100.0
    for index in range(length):
        if pullback[0] <= index < pullback[1]:
            price *= 0.996
        else:
            price *= 1.0 + drift
        closes.append(price)
    return closes


@pytest.fixture
def candle_factory():
    return make_candles


@pytest.fixture
def uptrend_candles():
    return make_candles(trending_closes())


@pytest.fixture
def flat_candles():
    return make_candles([100.0] * 700, wick=0.0002)
