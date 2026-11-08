from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest

from mtmbot.metrics import compute_performance
from mtmbot.models import Side, Trade

START = datetime(2024, 1, 1, tzinfo=UTC)


def _trade(pnl: float, r: float, hours: int = 1) -> Trade:
    return Trade(
        symbol="BTC/USD",
        side=Side.LONG,
        qty=1.0,
        entry_price=100.0,
        exit_price=100.0 + pnl,
        entry_ts=START,
        exit_ts=START + timedelta(hours=hours),
        fees=0.0,
        pnl=pnl,
        r_multiple=r,
        exit_reason="test",
    )


def _curve(values: list[float], step_hours: int = 1):
    return [(START + timedelta(hours=step_hours * i), value) for i, value in enumerate(values)]


def test_return_and_drawdown() -> None:
    perf = compute_performance(
        equity_curve=_curve([100.0, 110.0, 105.0, 120.0]),
        exposure=[0.0, 0.0, 0.0, 0.0],
        trades=[],
        step_seconds=3600,
        starting_equity=100.0,
        fees_paid=0.0,
    )
    assert perf.total_return_pct == pytest.approx(20.0)
    assert perf.max_drawdown_pct == pytest.approx(100 * 5 / 110)
    assert perf.trades == 0
    assert perf.win_rate_pct == 0.0
    assert perf.profit_factor == 0.0
    assert perf.max_drawdown_days >= 0.0


def test_constant_curve_has_zero_risk_stats() -> None:
    perf = compute_performance(
        equity_curve=_curve([100.0] * 10),
        exposure=[0.0] * 10,
        trades=[],
        step_seconds=3600,
        starting_equity=100.0,
        fees_paid=0.0,
    )
    assert perf.total_return_pct == pytest.approx(0.0)
    assert perf.sharpe == 0.0
    assert perf.sortino == 0.0
    assert perf.max_drawdown_pct == pytest.approx(0.0)
    assert perf.exposure_pct == pytest.approx(0.0)


def test_trade_statistics() -> None:
    trades = [_trade(10.0, 1.0), _trade(-5.0, -0.5), _trade(-5.0, -0.5), _trade(20.0, 2.0)]
    perf = compute_performance(
        equity_curve=_curve([100.0, 120.0]),
        exposure=[1.0, 1.0],
        trades=trades,
        step_seconds=3600,
        starting_equity=100.0,
        fees_paid=1.0,
    )
    assert perf.trades == 4
    assert perf.wins == 2
    assert perf.losses == 2
    assert perf.win_rate_pct == pytest.approx(50.0)
    assert perf.profit_factor == pytest.approx(30.0 / 10.0)
    assert perf.expectancy_r == pytest.approx(0.5)
    assert perf.best_r == pytest.approx(2.0)
    assert perf.worst_r == pytest.approx(-0.5)
    assert perf.max_consecutive_losses == 2
    assert perf.exposure_pct == pytest.approx(100.0)


def test_infinite_profit_factor_becomes_none() -> None:
    perf = compute_performance(
        equity_curve=_curve([100.0, 110.0]),
        exposure=[1.0, 1.0],
        trades=[_trade(10.0, 1.0)],
        step_seconds=3600,
        starting_equity=100.0,
        fees_paid=0.0,
    )
    assert math.isinf(perf.profit_factor)
    assert perf.to_dict()["profit_factor"] is None


def test_bad_inputs_raise() -> None:
    with pytest.raises(ValueError, match="empty"):
        compute_performance(
            equity_curve=[],
            exposure=[],
            trades=[],
            step_seconds=3600,
            starting_equity=100.0,
            fees_paid=0.0,
        )
    with pytest.raises(ValueError, match="same length"):
        compute_performance(
            equity_curve=_curve([100.0, 101.0]),
            exposure=[0.0],
            trades=[],
            step_seconds=3600,
            starting_equity=100.0,
            fees_paid=0.0,
        )
    with pytest.raises(ValueError, match="non-positive"):
        compute_performance(
            equity_curve=_curve([100.0, 0.0]),
            exposure=[0.0, 0.0],
            trades=[],
            step_seconds=3600,
            starting_equity=100.0,
            fees_paid=0.0,
        )
