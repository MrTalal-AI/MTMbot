"""Higher-timeframe aggregation and look-ahead-free alignment.

The strategy is only allowed to read a higher-timeframe bar once that bar has
*closed*. ``map_htf_indices`` enforces exactly that and is covered by tests.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

from mtmbot.models import Candle

__all__ = ["map_htf_indices", "resample"]


def resample(candles: Sequence[Candle], source_step: int, factor: int) -> list[Candle]:
    """Aggregate candles by an integer ``factor``, dropping incomplete buckets."""
    if source_step <= 0:
        raise ValueError(f"source_step must be positive, got {source_step!r}")
    if isinstance(factor, bool) or not isinstance(factor, int):
        raise TypeError(f"factor must be an int, got {type(factor).__name__}")
    if factor < 2:
        raise ValueError(f"factor must be >= 2, got {factor}")
    if not candles:
        return []

    target_step = source_step * factor
    buckets: dict[int, list[Candle]] = {}
    for candle in candles:
        start = int(candle.ts.timestamp())
        bucket = start - start % target_step
        buckets.setdefault(bucket, []).append(candle)

    last_complete = int(candles[-1].ts.timestamp()) + source_step
    out: list[Candle] = []
    for bucket in sorted(buckets):
        if bucket + target_step > last_complete:
            continue  # still forming, never exposed to the strategy
        group = buckets[bucket]
        out.append(
            Candle(
                ts=datetime.fromtimestamp(bucket, tz=UTC),
                open=group[0].open,
                high=max(candle.high for candle in group),
                low=min(candle.low for candle in group),
                close=group[-1].close,
                volume=sum(candle.volume for candle in group),
            )
        )
    return out


def map_htf_indices(
    candles: Sequence[Candle], htf_candles: Sequence[Candle], htf_step: int
) -> list[int]:
    """For each base bar, the index of the last *closed* higher-timeframe bar.

    Returns ``-1`` while no higher-timeframe bar has closed yet. A bar opening
    at ``t`` can only use higher-timeframe bars whose close time is ``<= t``,
    which is the definition of no look-ahead.
    """
    if htf_step <= 0:
        raise ValueError(f"htf_step must be positive, got {htf_step!r}")

    close_times = [int(candle.ts.timestamp()) + htf_step for candle in htf_candles]
    out: list[int] = []
    pointer = -1
    for candle in candles:
        start = int(candle.ts.timestamp())
        while pointer + 1 < len(close_times) and close_times[pointer + 1] <= start:
            pointer += 1
        out.append(pointer)
    return out
