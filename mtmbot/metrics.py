"""Performance statistics for an equity curve and a trade list.

Every number here is computed from the simulated equity curve, not from the
trades alone, so fees, slippage and idle time all affect the result.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from itertools import pairwise

from mtmbot.models import Trade
from mtmbot.timeframe import bars_per_year

__all__ = ["Performance", "compute_performance"]


def _finite(value: float) -> float:
    return value if math.isfinite(value) else 0.0


@dataclass(frozen=True, slots=True)
class Performance:
    """Headline statistics for a backtest or a live session."""

    total_return_pct: float
    cagr_pct: float
    sharpe: float
    sortino: float
    max_drawdown_pct: float
    max_drawdown_days: float
    calmar: float
    recovery_factor: float
    trades: int
    wins: int
    losses: int
    win_rate_pct: float
    profit_factor: float
    expectancy_r: float
    avg_win_r: float
    avg_loss_r: float
    best_r: float
    worst_r: float
    max_consecutive_losses: int
    avg_hold_hours: float
    exposure_pct: float
    fees_paid: float

    def to_dict(self) -> dict[str, float | int | None]:
        """JSON-safe view (infinite profit factors become ``None``)."""
        out: dict[str, float | int | None] = {
            "total_return_pct": self.total_return_pct,
            "cagr_pct": self.cagr_pct,
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "max_drawdown_pct": self.max_drawdown_pct,
            "max_drawdown_days": self.max_drawdown_days,
            "calmar": self.calmar,
            "recovery_factor": self.recovery_factor,
            "trades": self.trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate_pct": self.win_rate_pct,
            "profit_factor": self.profit_factor if math.isfinite(self.profit_factor) else None,
            "expectancy_r": self.expectancy_r,
            "avg_win_r": self.avg_win_r,
            "avg_loss_r": self.avg_loss_r,
            "best_r": self.best_r,
            "worst_r": self.worst_r,
            "max_consecutive_losses": self.max_consecutive_losses,
            "avg_hold_hours": self.avg_hold_hours,
            "exposure_pct": self.exposure_pct,
            "fees_paid": self.fees_paid,
        }
        return out


def _max_drawdown(values: Sequence[float], timestamps: Sequence[datetime]) -> tuple[float, float]:
    """Max drawdown in percent plus the longest peak-to-recovery in days."""
    peak = values[0]
    peak_ts = timestamps[0]
    worst_pct = 0.0
    worst_days = 0.0
    for value, ts in zip(values, timestamps, strict=True):
        if value >= peak:
            peak = value
            peak_ts = ts
            continue
        if peak > 0:
            worst_pct = max(worst_pct, 100.0 * (peak - value) / peak)
        worst_days = max(worst_days, (ts - peak_ts).total_seconds() / 86400.0)
    return worst_pct, worst_days


def _profit_factor(trades: Sequence[Trade]) -> float:
    gross_profit = sum(trade.pnl for trade in trades if trade.pnl > 0)
    gross_loss = -sum(trade.pnl for trade in trades if trade.pnl < 0)
    if gross_loss == 0:
        return math.inf if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def _streak(trades: Sequence[Trade]) -> int:
    worst = 0
    current = 0
    for trade in trades:
        if trade.pnl < 0:
            current += 1
            worst = max(worst, current)
        else:
            current = 0
    return worst


def compute_performance(
    *,
    equity_curve: Sequence[tuple[datetime, float]],
    exposure: Sequence[float],
    trades: Sequence[Trade],
    step_seconds: int,
    starting_equity: float,
    fees_paid: float,
) -> Performance:
    """Compute all statistics from the equity curve.

    Raises:
        ValueError: if the curve is empty or the starting equity is invalid.
    """
    if not equity_curve:
        raise ValueError("cannot compute performance of an empty equity curve")
    if starting_equity <= 0:
        raise ValueError(f"starting_equity must be > 0, got {starting_equity!r}")
    if len(equity_curve) != len(exposure):
        raise ValueError("equity_curve and exposure must have the same length")

    timestamps = [point[0] for point in equity_curve]
    values = [float(point[1]) for point in equity_curve]
    if any(value <= 0 for value in values):
        raise ValueError("equity curve contains a non-positive value")
    if any(later <= earlier for earlier, later in pairwise(timestamps)):
        raise ValueError("equity curve timestamps must be strictly increasing")

    total_return_pct = 100.0 * (values[-1] / starting_equity - 1.0)

    span_seconds = (timestamps[-1] - timestamps[0]).total_seconds() + step_seconds
    years = span_seconds / (365.0 * 86400.0)
    if years >= 1.0 / 365.0 and values[-1] > 0:
        cagr_pct = 100.0 * ((values[-1] / starting_equity) ** (1.0 / years) - 1.0)
    else:
        cagr_pct = total_return_pct

    returns: list[float] = []
    for previous, current in pairwise(values):
        if previous > 0:
            returns.append(current / previous - 1.0)

    if len(returns) > 1:
        mean = sum(returns) / len(returns)
        variance = sum((value - mean) ** 2 for value in returns) / (len(returns) - 1)
        std = math.sqrt(variance)
        downside = math.sqrt(sum(min(value, 0.0) ** 2 for value in returns) / len(returns))
        annualiser = math.sqrt(bars_per_year(step_seconds))
        sharpe = (mean / std) * annualiser if std > 0 else 0.0
        sortino = (mean / downside) * annualiser if downside > 0 else 0.0
    else:
        sharpe = 0.0
        sortino = 0.0

    max_drawdown_pct, max_drawdown_days = _max_drawdown(values, timestamps)
    calmar = cagr_pct / max_drawdown_pct if max_drawdown_pct > 0 else 0.0
    recovery = total_return_pct / max_drawdown_pct if max_drawdown_pct > 0 else 0.0

    wins = [trade for trade in trades if trade.pnl > 0]
    losses = [trade for trade in trades if trade.pnl <= 0]
    trade_count = len(trades)
    win_rate_pct = 100.0 * len(wins) / trade_count if trade_count else 0.0
    r_values = [trade.r_multiple for trade in trades]
    avg_win_r = sum(trade.r_multiple for trade in wins) / len(wins) if wins else 0.0
    avg_loss_r = sum(trade.r_multiple for trade in losses) / len(losses) if losses else 0.0
    total_r = sum(r_values)
    exposure_pct = 100.0 * sum(1 for value in exposure if value > 0) / len(exposure)

    return Performance(
        total_return_pct=_finite(total_return_pct),
        cagr_pct=_finite(cagr_pct),
        sharpe=_finite(sharpe),
        sortino=_finite(sortino),
        max_drawdown_pct=_finite(max_drawdown_pct),
        max_drawdown_days=_finite(max_drawdown_days),
        calmar=_finite(calmar),
        recovery_factor=_finite(recovery),
        trades=trade_count,
        wins=len(wins),
        losses=len(losses),
        win_rate_pct=_finite(win_rate_pct),
        profit_factor=_profit_factor(trades),
        expectancy_r=_finite(total_r / trade_count) if trade_count else 0.0,
        avg_win_r=_finite(avg_win_r),
        avg_loss_r=_finite(avg_loss_r),
        best_r=_finite(max(r_values)) if r_values else 0.0,
        worst_r=_finite(min(r_values)) if r_values else 0.0,
        max_consecutive_losses=_streak(trades),
        avg_hold_hours=_finite(
            sum(trade.holding_seconds for trade in trades) / trade_count / 3600.0
        )
        if trade_count
        else 0.0,
        exposure_pct=_finite(exposure_pct),
        fees_paid=_finite(fees_paid),
    )
