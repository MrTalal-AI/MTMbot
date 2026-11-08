from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from mtmbot.config import BotConfig
from mtmbot.execution import PaperBroker, Portfolio
from mtmbot.models import Side, Signal
from mtmbot.risk import RiskManager
from mtmbot.session import TradingSession
from mtmbot.state import StateError, restore_state, save_state
from mtmbot.strategy import MTMStrategy

NOW = datetime(2024, 1, 1, tzinfo=UTC)


def _session(symbol: str = "BTC/USD") -> TradingSession:
    config = BotConfig(symbol=symbol)
    return TradingSession(
        symbol=symbol,
        portfolio=Portfolio.create(10_000.0),
        risk=RiskManager(config.risk, 10_000.0),
        broker=PaperBroker(costs=config.costs),
        strategy=MTMStrategy(config.strategy, config.step_seconds),
    )


def test_state_roundtrip(tmp_path) -> None:
    session = _session()
    session.portfolio.open_position(
        symbol="BTC/USD",
        side=Side.LONG,
        qty=0.5,
        price=100.0,
        stop_price=98.0,
        take_profit_price=104.0,
        ts=NOW,
        fee=1.0,
        reason="test entry",
    )
    session.portfolio.close_position(symbol="BTC/USD", price=103.0, ts=NOW, fee=0.5)
    session.portfolio.open_position(
        symbol="BTC/USD",
        side=Side.LONG,
        qty=0.25,
        price=105.0,
        stop_price=102.0,
        take_profit_price=None,
        ts=NOW + timedelta(hours=1),
        fee=0.25,
        reason="second entry",
    )
    session.pending = Signal.enter_long(99.0, "queued", take_profit_price=103.0)
    session.risk.update(9_500.0, NOW)

    path = tmp_path / "state.json"
    last_bar = NOW + timedelta(hours=5)
    save_state(path, session, last_bar)

    restored = _session()
    loaded_last = restore_state(path, restored)

    assert loaded_last == last_bar
    assert restored.portfolio.cash == pytest.approx(session.portfolio.cash)
    assert restored.portfolio.fees_paid == pytest.approx(session.portfolio.fees_paid)
    assert restored.portfolio.realized_pnl == pytest.approx(session.portfolio.realized_pnl)
    assert len(restored.portfolio.trades) == 1
    assert restored.portfolio.trades[0].pnl == pytest.approx(session.portfolio.trades[0].pnl)
    position = restored.portfolio.positions["BTC/USD"]
    assert position.qty == pytest.approx(0.25)
    assert position.entry_price == pytest.approx(105.0)
    assert position.stop_price == pytest.approx(102.0)
    assert restored.pending is not None
    assert restored.pending.stop_price == pytest.approx(99.0)
    assert restored.risk.peak_equity == pytest.approx(10_000.0)


def test_missing_state_is_not_an_error(tmp_path) -> None:
    assert restore_state(tmp_path / "absent.json", _session()) is None


def test_corrupt_state_raises(tmp_path) -> None:
    path = tmp_path / "state.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(StateError):
        restore_state(path, _session())


def test_symbol_mismatch_raises(tmp_path) -> None:
    path = tmp_path / "state.json"
    save_state(path, _session("BTC/USD"), NOW)
    with pytest.raises(StateError, match="trades"):
        restore_state(path, _session("ETH/USD"))


def test_save_is_atomic(tmp_path) -> None:
    path = tmp_path / "state.json"
    save_state(path, _session(), NOW)
    assert path.exists()
    assert list(tmp_path.glob("*.tmp")) == []
