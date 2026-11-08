"""Backtesting: replay history through the exact trading code path.

Guarantees:

* a decision made on bar ``i`` is executed at bar ``i + 1``'s open;
* stops and take profits are checked inside each bar, with the stop winning
  any tie;
* fees and slippage are charged on every fill;
* the run is deterministic, so identical inputs produce identical results.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from itertools import pairwise
from typing import Any

from mtmbot.config import BotConfig
from mtmbot.execution import PaperBroker, Portfolio
from mtmbot.metrics import Performance, compute_performance
from mtmbot.models import Candle, Trade
from mtmbot.risk import RiskManager
from mtmbot.session import TradingSession
from mtmbot.strategy.mtm import MTMStrategy

__all__ = ["BacktestError", "BacktestResult", "EquityPoint", "run_backtest"]


class BacktestError(RuntimeError):
    """Raised when a backtest cannot be run meaningfully."""


@dataclass(frozen=True, slots=True)
class EquityPoint:
    """One mark-to-market observation."""

    ts: datetime
    equity: float
    cash: float
    exposure: float
    price: float


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Everything a report needs, frozen so it cannot drift after the run."""

    symbol: str
    timeframe_seconds: int
    bars: int
    htf_bars: int
    starting_equity: float
    ending_equity: float
    equity_curve: tuple[EquityPoint, ...]
    trades: tuple[Trade, ...]
    performance: Performance
    buy_hold_return_pct: float
    open_position_at_end: bool
    data_gaps: int
    config_summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe summary of the run."""
        return {
            "symbol": self.symbol,
            "timeframe_seconds": self.timeframe_seconds,
            "bars": self.bars,
            "htf_bars": self.htf_bars,
            "starting_equity": self.starting_equity,
            "ending_equity": self.ending_equity,
            "buy_hold_return_pct": self.buy_hold_return_pct,
            "open_position_at_end": self.open_position_at_end,
            "data_gaps": self.data_gaps,
            "performance": self.performance.to_dict(),
            "trades": [
                {
                    "side": trade.side.value,
                    "qty": trade.qty,
                    "entry_ts": trade.entry_ts.isoformat(),
                    "exit_ts": trade.exit_ts.isoformat(),
                    "entry_price": trade.entry_price,
                    "exit_price": trade.exit_price,
                    "pnl": trade.pnl,
                    "r_multiple": trade.r_multiple,
                    "fees": trade.fees,
                    "exit_reason": trade.exit_reason,
                }
                for trade in self.trades
            ],
            "config": self.config_summary,
        }


def _count_gaps(candles: Sequence[Candle], step_seconds: int) -> int:
    gaps = 0
    for previous, current in pairwise(candles):
        delta = (current.ts - previous.ts).total_seconds()
        if delta > step_seconds * 1.5:
            gaps += 1
    return gaps


def run_backtest(config: BotConfig, candles: Sequence[Candle]) -> BacktestResult:
    """Replay ``candles`` and return a full result.

    Raises:
        BacktestError: if there is not enough history to warm the strategy up.
    """
    if not isinstance(config, BotConfig):
        raise TypeError(f"config must be a BotConfig, got {type(config).__name__}")
    series = list(candles)
    if len(series) < 2:
        raise BacktestError("need at least 2 candles to backtest")
    for previous, current in pairwise(series):
        if current.ts <= previous.ts:
            raise BacktestError("candles must be sorted by open time with no duplicates")
        delta = int((current.ts - previous.ts).total_seconds())
        if delta % config.step_seconds != 0:
            raise BacktestError(
                f"candle spacing is {delta}s but the config timeframe is "
                f"{config.timeframe} ({config.step_seconds}s); download the data at the "
                "configured timeframe instead of mixing intervals"
            )

    strategy = MTMStrategy(config.strategy, config.step_seconds)
    needed = strategy.warmup_bars
    if len(series) < needed:
        raise BacktestError(
            f"not enough history: {len(series)} bars, strategy needs {needed} for warm-up. "
            "Download a longer range."
        )

    prepared = strategy.prepare_from_base(series)
    portfolio = Portfolio.create(config.starting_cash)
    session = TradingSession(
        symbol=config.symbol,
        portfolio=portfolio,
        risk=RiskManager(config.risk, config.starting_cash),
        broker=PaperBroker(costs=config.costs),
        strategy=strategy,
    )

    curve: list[EquityPoint] = []
    exposure: list[float] = []
    marks = {config.symbol: series[0].open}
    for index, candle in enumerate(series):
        marks[config.symbol] = candle.open
        session.begin_bar(candle.open, candle.ts)
        session.end_bar(candle, prepared, index)
        marks[config.symbol] = candle.close
        equity = portfolio.equity(marks)
        held = portfolio.exposure(marks)
        curve.append(
            EquityPoint(
                ts=candle.ts,
                equity=equity,
                cash=portfolio.cash,
                exposure=held,
                price=candle.close,
            )
        )
        exposure.append(held)

    performance = compute_performance(
        equity_curve=[(point.ts, point.equity) for point in curve],
        exposure=exposure,
        trades=portfolio.trades,
        step_seconds=config.step_seconds,
        starting_equity=config.starting_cash,
        fees_paid=portfolio.fees_paid,
    )

    buy_hold = 100.0 * (series[-1].close / series[0].open - 1.0)
    return BacktestResult(
        symbol=config.symbol,
        timeframe_seconds=config.step_seconds,
        bars=len(series),
        htf_bars=prepared.htf_bars,
        starting_equity=config.starting_cash,
        ending_equity=curve[-1].equity,
        equity_curve=tuple(curve),
        trades=tuple(portfolio.trades),
        performance=performance,
        buy_hold_return_pct=buy_hold,
        open_position_at_end=bool(portfolio.positions),
        data_gaps=_count_gaps(series, config.step_seconds),
        config_summary=config.summary(),
    )
