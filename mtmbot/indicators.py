"""Causal technical indicators implemented without NumPy.

Every function returns a list aligned with the input, using ``None`` for values
that are not yet defined (warm-up). A value at index ``i`` is guaranteed to
depend only on inputs at indices ``<= i``; that property is what keeps the
backtest free of look-ahead bias and is covered by tests.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence

from mtmbot.models import Candle

__all__ = [
    "atr",
    "crossover",
    "crossunder",
    "ema",
    "rolling_max",
    "rolling_min",
    "rolling_std",
    "rsi",
    "sma",
    "true_range",
]


def _check_period(period: int) -> int:
    if isinstance(period, bool) or not isinstance(period, int):
        raise TypeError(f"period must be an int, got {type(period).__name__}")
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    return period


def _check_series(values: Sequence[float], name: str = "values") -> Sequence[float]:
    if not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence, got {type(values).__name__}")
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name}[{index}] must be a number, got {type(value).__name__}")
        if not math.isfinite(float(value)):
            raise ValueError(f"{name}[{index}] must be finite, got {value!r}")
    return values


def sma(values: Sequence[float], period: int) -> list[float | None]:
    """Simple moving average."""
    period = _check_period(period)
    _check_series(values)
    out: list[float | None] = [None] * len(values)
    window = 0.0
    for i, value in enumerate(values):
        window += float(value)
        if i >= period:
            window -= float(values[i - period])
        if i >= period - 1:
            out[i] = window / period
    return out


def ema(values: Sequence[float], period: int) -> list[float | None]:
    """Exponential moving average, seeded with the SMA of the first window."""
    period = _check_period(period)
    _check_series(values)
    out: list[float | None] = [None] * len(values)
    if len(values) < period:
        return out
    seed = sum(float(v) for v in values[:period]) / period
    out[period - 1] = seed
    alpha = 2.0 / (period + 1.0)
    previous = seed
    for i in range(period, len(values)):
        previous = float(values[i]) * alpha + previous * (1.0 - alpha)
        out[i] = previous
    return out


def _rsi_point(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0.0:
        return 100.0 if avg_gain > 0.0 else 50.0
    if avg_gain == 0.0:
        return 0.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def rsi(values: Sequence[float], period: int = 14) -> list[float | None]:
    """Wilder's Relative Strength Index."""
    period = _check_period(period)
    _check_series(values)
    out: list[float | None] = [None] * len(values)
    if len(values) <= period:
        return out
    gains = 0.0
    losses = 0.0
    for i in range(1, period + 1):
        change = float(values[i]) - float(values[i - 1])
        if change >= 0:
            gains += change
        else:
            losses -= change
    avg_gain = gains / period
    avg_loss = losses / period
    out[period] = _rsi_point(avg_gain, avg_loss)
    for i in range(period + 1, len(values)):
        change = float(values[i]) - float(values[i - 1])
        gain = max(change, 0.0)
        loss = max(-change, 0.0)
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        out[i] = _rsi_point(avg_gain, avg_loss)
    return out


def true_range(candles: Sequence[Candle]) -> list[float]:
    """True range of each bar (first bar uses high-low)."""
    out: list[float] = []
    previous_close: float | None = None
    for candle in candles:
        if previous_close is None:
            out.append(candle.high - candle.low)
        else:
            out.append(
                max(
                    candle.high - candle.low,
                    abs(candle.high - previous_close),
                    abs(candle.low - previous_close),
                )
            )
        previous_close = candle.close
    return out


def atr(candles: Sequence[Candle], period: int = 14) -> list[float | None]:
    """Wilder's Average True Range."""
    period = _check_period(period)
    out: list[float | None] = [None] * len(candles)
    ranges = true_range(candles)
    if len(candles) < period + 1:
        return out
    previous = sum(ranges[: period + 1]) / (period + 1)
    out[period] = previous
    for i in range(period + 1, len(candles)):
        previous = (previous * (period - 1) + ranges[i]) / period
        out[i] = previous
    return out


def rolling_max(values: Sequence[float], period: int) -> list[float | None]:
    """Highest value in the trailing ``period`` window (inclusive of ``i``)."""
    period = _check_period(period)
    _check_series(values)
    out: list[float | None] = [None] * len(values)
    window: deque[int] = deque()
    for i, value in enumerate(values):
        current = float(value)
        while window and float(values[window[-1]]) <= current:
            window.pop()
        window.append(i)
        if window[0] <= i - period:
            window.popleft()
        if i >= period - 1:
            out[i] = float(values[window[0]])
    return out


def rolling_min(values: Sequence[float], period: int) -> list[float | None]:
    """Lowest value in the trailing ``period`` window (inclusive of ``i``)."""
    period = _check_period(period)
    _check_series(values)
    out: list[float | None] = [None] * len(values)
    window: deque[int] = deque()
    for i, value in enumerate(values):
        current = float(value)
        while window and float(values[window[-1]]) >= current:
            window.pop()
        window.append(i)
        if window[0] <= i - period:
            window.popleft()
        if i >= period - 1:
            out[i] = float(values[window[0]])
    return out


def rolling_std(values: Sequence[float], period: int) -> list[float | None]:
    """Population standard deviation of the trailing window."""
    period = _check_period(period)
    _check_series(values)
    out: list[float | None] = [None] * len(values)
    window: deque[float] = deque()
    total = 0.0
    total_sq = 0.0
    for i, value in enumerate(values):
        current = float(value)
        window.append(current)
        total += current
        total_sq += current * current
        if len(window) > period:
            dropped = window.popleft()
            total -= dropped
            total_sq -= dropped * dropped
        if len(window) == period:
            mean = total / period
            variance = max(total_sq / period - mean * mean, 0.0)
            out[i] = math.sqrt(variance)
    return out


def crossover(fast: Sequence[float | None], slow: Sequence[float | None], index: int) -> bool:
    """True when ``fast`` crosses above ``slow`` exactly at ``index``."""
    if index < 1 or index >= min(len(fast), len(slow)):
        return False
    f_now, s_now = fast[index], slow[index]
    f_prev, s_prev = fast[index - 1], slow[index - 1]
    if f_now is None or s_now is None or f_prev is None or s_prev is None:
        return False
    return f_prev <= s_prev and f_now > s_now


def crossunder(fast: Sequence[float | None], slow: Sequence[float | None], index: int) -> bool:
    """True when ``fast`` crosses below ``slow`` exactly at ``index``."""
    if index < 1 or index >= min(len(fast), len(slow)):
        return False
    f_now, s_now = fast[index], slow[index]
    f_prev, s_prev = fast[index - 1], slow[index - 1]
    if f_now is None or s_now is None or f_prev is None or s_prev is None:
        return False
    return f_prev >= s_prev and f_now < s_now
