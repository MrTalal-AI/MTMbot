from __future__ import annotations

from datetime import UTC, datetime

import pytest

from mtmbot.backtest import BacktestError, run_backtest
from mtmbot.config import BotConfig, RiskConfig, StrategyConfig
from mtmbot.execution import PaperBroker, Portfolio
from mtmbot.models import Signal
from mtmbot.risk import RiskManager
from mtmbot.session import TradingSession
from mtmbot.strategy import MTMStrategy


def _config(**risk_overrides: object) -> BotConfig:
    risk: dict[str, object] = {
        "risk_per_trade_pct": 1.0,
        "daily_loss_limit_pct": 20.0,
        "max_drawdown_pct": 50.0,
        "min_notional": 1.0,
    }
    risk.update(risk_overrides)
    return BotConfig(
        symbol="BTC/USD",
        timeframe="1h",
        starting_cash=10_000.0,
        strategy=StrategyConfig(
            fast_ema=10,
            slow_ema=30,
            rsi_period=14,
            atr_period=14,
            htf_factor=4,
            atr_stop_mult=2.0,
            reward_risk=2.0,
            rsi_long_min=52.0,
            rsi_short_max=48.0,
        ),
        risk=RiskConfig(**risk),  # type: ignore[arg-type]
    )


def test_flat_market_never_trades(flat_candles) -> None:
    result = run_backtest(_config(), flat_candles)
    assert result.performance.trades == 0
    assert result.ending_equity == pytest.approx(10_000.0)
    assert result.ending_equity == pytest.approx(result.starting_equity)
    assert result.performance.total_return_pct == pytest.approx(0.0)
    assert not result.open_position_at_end


def test_uptrend_backtest_is_consistent(uptrend_candles) -> None:
    result = run_backtest(_config(), uptrend_candles)
    assert result.bars == len(uptrend_candles)
    assert len(result.equity_curve) == len(uptrend_candles)
    assert result.performance.trades >= 1
    assert result.ending_equity == pytest.approx(result.equity_curve[-1].equity)
    assert result.performance.max_drawdown_pct >= 0
    assert result.performance.fees_paid > 0


def test_entries_are_filled_at_the_next_bar_open(uptrend_candles) -> None:
    config = _config()
    result = run_backtest(config, uptrend_candles)
    slip = config.costs.slippage_rate
    by_ts = {candle.ts: candle for candle in uptrend_candles}
    checked = 0
    for trade in result.trades:
        if trade.exit_reason == "invalidated entry":
            continue
        candle = by_ts[trade.entry_ts]
        expected = candle.open * (1 + slip) if trade.side.value == "long" else candle.open * (
            1 - slip
        )
        assert trade.entry_price == pytest.approx(expected, rel=1e-12)
        checked += 1
    assert checked > 0


def test_backtest_is_deterministic(uptrend_candles) -> None:
    first = run_backtest(_config(), uptrend_candles)
    second = run_backtest(_config(), uptrend_candles)
    assert first.ending_equity == second.ending_equity
    assert [(t.entry_ts, t.exit_ts, t.pnl) for t in first.trades] == [
        (t.entry_ts, t.exit_ts, t.pnl) for t in second.trades
    ]


def test_no_lookahead_prefix_matches(uptrend_candles) -> None:
    """Processing only past data must reproduce the same past equity."""
    full = run_backtest(_config(), uptrend_candles)
    truncated = run_backtest(_config(), uptrend_candles[:400])
    assert [point.equity for point in truncated.equity_curve] == pytest.approx(
        [point.equity for point in full.equity_curve[:400]]
    )
    assert [point.ts for point in truncated.equity_curve] == [
        point.ts for point in full.equity_curve[:400]
    ]


def test_risk_limits_cap_position_size(uptrend_candles) -> None:
    config = _config(risk_per_trade_pct=0.1, max_notional_pct=10.0)
    result = run_backtest(config, uptrend_candles)
    for trade in result.trades:
        notional = trade.entry_price * trade.qty
        assert notional <= config.starting_cash * 0.10 * 3 + 1e-6


def test_kill_switch_blocks_entries_in_the_session() -> None:
    config = _config(max_drawdown_pct=2.0, daily_loss_limit_pct=3.0)
    portfolio = Portfolio.create(10_000.0)
    risk = RiskManager(config.risk, 10_000.0)
    session = TradingSession(
        symbol=config.symbol,
        portfolio=portfolio,
        risk=risk,
        broker=PaperBroker(costs=config.costs),
        strategy=MTMStrategy(config.strategy, config.step_seconds),
    )
    moment = datetime(2024, 1, 1, tzinfo=UTC)
    risk.update(10_000.0, moment)
    risk.update(9_000.0, moment)  # 10% drawdown, limit is 2%
    assert risk.halted

    session.pending = Signal.enter_long(99.0, "test entry", take_profit_price=103.0)
    session.begin_bar(100.0, moment)
    assert portfolio.positions == {}
    assert portfolio.trades == []


def test_too_little_data_raises(flat_candles) -> None:
    with pytest.raises(BacktestError, match="not enough history"):
        run_backtest(_config(), flat_candles[:50])


def test_timeframe_mismatch_raises(flat_candles) -> None:
    from dataclasses import replace

    hourly = _config()
    config = replace(hourly, timeframe="4h")
    with pytest.raises(BacktestError, match="candle spacing"):
        run_backtest(config, flat_candles)


def test_unsorted_data_raises(flat_candles) -> None:
    broken = list(flat_candles[:200])
    broken[10], broken[11] = broken[11], broken[10]
    with pytest.raises(BacktestError, match="sorted"):
        run_backtest(_config(), broken)


def test_to_dict_is_json_safe(uptrend_candles) -> None:
    import json

    payload = run_backtest(_config(), uptrend_candles).to_dict()
    text = json.dumps(payload, allow_nan=False)
    assert '"performance"' in text
    assert payload["bars"] == len(uptrend_candles)
