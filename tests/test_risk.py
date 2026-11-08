from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from mtmbot.config import RiskConfig
from mtmbot.models import Side
from mtmbot.risk import RiskManager

START = datetime(2024, 1, 1, tzinfo=UTC)


def _manager(**overrides: object) -> RiskManager:
    payload: dict[str, object] = {
        "risk_per_trade_pct": 1.0,
        "max_positions": 1,
        "max_notional_pct": 500.0,
        "daily_loss_limit_pct": 3.0,
        "max_drawdown_pct": 20.0,
        "min_notional": 1.0,
        "max_stop_distance_pct": 10.0,
    }
    payload.update(overrides)
    return RiskManager(RiskConfig(**payload), 10_000.0)  # type: ignore[arg-type]


def test_sizing_uses_stop_distance() -> None:
    decision = _manager().size(10_000.0, 100.0, 98.0)
    assert decision.allowed
    assert decision.qty == pytest.approx(50.0)
    assert decision.risk_amount == pytest.approx(100.0)
    assert decision.notional == pytest.approx(5_000.0)


def test_notional_cap_limits_size() -> None:
    manager = _manager(max_notional_pct=50.0)
    decision = manager.size(10_000.0, 100.0, 99.99)
    assert decision.allowed
    assert decision.notional <= 5_000.0 + 1e-6


def test_min_notional_rejects_dust() -> None:
    manager = _manager(min_notional=1_000.0)
    decision = manager.size(10.0, 100.0, 98.0)
    assert not decision.allowed
    assert "minimum" in decision.reason


def test_stop_on_wrong_side_is_rejected() -> None:
    manager = _manager()
    decision = manager.size(10_000.0, 100.0, 101.0, side=Side.LONG)
    assert not decision.allowed
    assert "not below entry" in decision.reason


def test_stop_too_wide_is_rejected() -> None:
    manager = _manager(max_stop_distance_pct=2.0)
    decision = manager.size(10_000.0, 100.0, 90.0)
    assert not decision.allowed
    assert "wider than" in decision.reason


def test_daily_loss_halts_and_resets_next_day() -> None:
    manager = _manager()
    manager.update(10_000.0, START)
    manager.update(9_600.0, START + timedelta(hours=2))
    assert manager.halted
    assert "daily loss" in (manager.halt_reason or "")

    manager.update(9_600.0, START + timedelta(days=1))
    assert not manager.halted


def test_drawdown_halt_is_sticky() -> None:
    manager = _manager(max_drawdown_pct=5.0, daily_loss_limit_pct=10.0)
    manager.update(10_000.0, START)
    manager.update(9_400.0, START + timedelta(hours=1))
    assert manager.halted
    assert "drawdown" in (manager.halt_reason or "")

    manager.update(9_400.0, START + timedelta(days=1))
    assert manager.halted


def test_halt_blocks_sizing() -> None:
    manager = _manager()
    manager.update(10_000.0, START)
    manager.update(9_000.0, START + timedelta(hours=1))
    assert manager.halted
    decision = manager.size(9_000.0, 100.0, 98.0)
    assert not decision.allowed


def test_can_open_respects_position_limit() -> None:
    manager = _manager()
    assert manager.can_open(0, 10_000.0)
    assert not manager.can_open(1, 10_000.0)


def test_snapshot_roundtrip() -> None:
    manager = _manager(max_drawdown_pct=5.0, daily_loss_limit_pct=10.0)
    manager.update(10_500.0, START)
    manager.update(9_900.0, START + timedelta(hours=1))
    snapshot = manager.snapshot()

    restored = _manager(max_drawdown_pct=5.0, daily_loss_limit_pct=10.0)
    restored.restore(snapshot)
    assert restored.halted
    assert restored.peak_equity == pytest.approx(10_500.0)


def test_manual_reset_clears_halt() -> None:
    manager = _manager()
    manager.update(10_000.0, START)
    manager.update(9_000.0, START + timedelta(hours=1))
    manager.reset_halt()
    assert not manager.halted
