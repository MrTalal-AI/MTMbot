"""CSV market data: the canonical on-disk format for backtests."""

from __future__ import annotations

import csv
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from mtmbot.data.base import DataError
from mtmbot.models import Candle

__all__ = ["read_candles", "write_candles"]

HEADER = ("ts", "open", "high", "low", "close", "volume")


def write_candles(path: str | Path, candles: list[Candle]) -> Path:
    """Write candles to CSV atomically (never leave a half-written dataset)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not candles:
        raise DataError("refusing to write an empty candle series")

    ordered = sorted(candles, key=lambda c: c.ts)
    handle, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=".candles-", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(HEADER)
            for candle in ordered:
                writer.writerow(
                    [
                        candle.ts.astimezone(UTC).isoformat(),
                        f"{candle.open:.10g}",
                        f"{candle.high:.10g}",
                        f"{candle.low:.10g}",
                        f"{candle.close:.10g}",
                        f"{candle.volume:.10g}",
                    ]
                )
        Path(tmp_name).replace(target)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return target


def _parse_ts(value: str, line: int) -> datetime:
    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise DataError(f"line {line}: invalid timestamp {value!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _parse_float(row: dict[str, str], key: str, line: int) -> float:
    raw = (row.get(key) or "").strip()
    if not raw:
        raise DataError(f"line {line}: missing column {key!r}")
    try:
        return float(raw)
    except ValueError as exc:
        raise DataError(f"line {line}: column {key!r} is not a number: {raw!r}") from exc


def read_candles(path: str | Path) -> list[Candle]:
    """Read and validate a candle CSV, returning unique bars sorted by time."""
    target = Path(path)
    if not target.is_file():
        raise DataError(f"data file not found: {target}")

    candles: list[Candle] = []
    with target.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise DataError(f"{target} is empty")
        missing = [column for column in HEADER if column not in reader.fieldnames]
        if missing:
            raise DataError(f"{target} is missing column(s): {', '.join(missing)}")
        for line, row in enumerate(reader, start=2):
            ts = _parse_ts(row["ts"], line)
            try:
                candle = Candle(
                    ts=ts,
                    open=_parse_float(row, "open", line),
                    high=_parse_float(row, "high", line),
                    low=_parse_float(row, "low", line),
                    close=_parse_float(row, "close", line),
                    volume=_parse_float(row, "volume", line),
                )
            except (ValueError, TypeError) as exc:
                raise DataError(f"{target} line {line}: {exc}") from exc
            candles.append(candle)

    if not candles:
        raise DataError(f"{target} contains no rows")

    by_ts: dict[datetime, Candle] = {}
    for candle in candles:
        by_ts[candle.ts] = candle
    return [by_ts[key] for key in sorted(by_ts)]
