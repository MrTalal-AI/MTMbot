from __future__ import annotations

import math

import pytest

from mtmbot.indicators import (
    atr,
    crossover,
    crossunder,
    ema,
    rolling_max,
    rolling_min,
    rolling_std,
    rsi,
    sma,
)


def test_sma() -> None:
    assert sma([1, 2, 3, 4, 5], 3) == [None, None, 2.0, 3.0, 4.0]


def test_sma_short_series() -> None:
    assert sma([1, 2], 3) == [None, None]


def test_ema_seeded_with_sma() -> None:
    out = ema([1, 2, 3, 4, 5], 3)
    assert out[:2] == [None, None]
    assert out[2] == pytest.approx(2.0)
    assert out[3] == pytest.approx(3.0)
    assert out[4] == pytest.approx(4.0)


def test_rsi_monotonic_is_100() -> None:
    values = [float(i) for i in range(1, 40)]
    out = rsi(values, 14)
    assert out[13] is None
    assert out[14] == pytest.approx(100.0)
    assert out[-1] == pytest.approx(100.0)


def test_rsi_flat_is_50() -> None:
    out = rsi([100.0] * 40, 14)
    assert out[-1] == pytest.approx(50.0)


def test_rsi_monotonic_down_is_zero() -> None:
    out = rsi([float(100 - i) for i in range(40)], 14)
    assert out[-1] == pytest.approx(0.0)


def test_rsi_rejects_bad_period() -> None:
    with pytest.raises(ValueError):
        rsi([1.0, 2.0], 0)


def test_atr_constant_range(candle_factory) -> None:
    candles = candle_factory([100.0] * 40, wick=0.01)
    out = atr(candles, 14)
    assert out[13] is None
    expected = 100.0 * 0.02  # high-low of a flat bar with 1% wicks
    assert out[14] == pytest.approx(expected, rel=1e-6)
    assert out[-1] == pytest.approx(expected, rel=1e-6)


def test_atr_captures_gaps(candle_factory) -> None:
    candles = candle_factory([100.0, 100.0, 130.0, 130.0])
    out = atr(candles, 2)
    assert out[-1] is not None and out[-1] > 3


def test_indicators_are_causal() -> None:
    """Adding future data must never change a past value."""
    values = [100 + math.sin(i / 3.0) * 5 for i in range(60)]
    full = ema(values, 10)
    partial = ema(values[:30], 10)
    assert partial == full[:30]

    rsi_full = rsi(values, 14)
    rsi_partial = rsi(values[:30], 14)
    assert rsi_partial == rsi_full[:30]


def test_rolling_extremes() -> None:
    values = [1.0, 3.0, 2.0, 5.0, 4.0]
    assert rolling_max(values, 2) == [None, 3.0, 3.0, 5.0, 5.0]
    assert rolling_min(values, 2) == [None, 1.0, 2.0, 2.0, 4.0]


def test_rolling_std() -> None:
    assert rolling_std([5.0] * 6, 3) == [None, None, 0.0, 0.0, 0.0, 0.0]


def test_crossover_and_crossunder() -> None:
    fast = [1.0, 2.0, 3.0, 4.0]
    slow = [2.0, 2.0, 2.0, 2.0]
    assert not crossover(fast, slow, 0)
    assert not crossover(fast, slow, 1)
    assert crossover(fast, slow, 2)
    assert not crossover(fast, slow, 3)
    assert crossunder(slow, fast, 2)


def test_crossover_handles_none() -> None:
    assert not crossover([None, 1.0], [1.0, 1.0], 1)


def test_indicators_reject_nan_input() -> None:
    with pytest.raises(ValueError):
        sma([1.0, float("inf")], 2)


def test_indicators_reject_non_sequence() -> None:
    with pytest.raises(TypeError):
        sma(42, 2)  # type: ignore[arg-type]
