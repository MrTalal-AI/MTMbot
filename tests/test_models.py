from __future__ import annotations

from datetime import UTC, datetime

import pytest

from mtmbot.models import Candle, Fill, Order, OrderSide, Position, Side, Signal, Trade


def _candle(**overrides: object) -> Candle:
    payload: dict[str, object] = {
        "ts": datetime(2024, 1, 1, tzinfo=UTC),
        "open": 100.0,
        "high": 102.0,
        "low": 99.0,
        "close": 101.0,
        "volume": 10.0,
    }
    payload.update(overrides)
    return Candle(**payload)  # type: ignore[arg-type]


def test_candle_ok() -> None:
    candle = _candle()
    assert candle.range == 3.0
    assert candle.is_bullish


def test_candle_rejects_naive_timestamp() -> None:
    with pytest.raises(ValueError):
        _candle(ts=datetime(2024, 1, 1))


def test_candle_normalises_timezone() -> None:
    candle = _candle(ts=datetime(2024, 1, 1, 3, tzinfo=UTC))
    assert candle.ts.tzinfo is UTC


def test_candle_rejects_high_below_low() -> None:
    with pytest.raises(ValueError):
        _candle(high=98.0)


def test_candle_rejects_close_outside_range() -> None:
    with pytest.raises(ValueError):
        _candle(close=200.0)


def test_candle_rejects_non_positive_price() -> None:
    with pytest.raises(ValueError):
        _candle(open=0.0)


def test_candle_rejects_nan() -> None:
    with pytest.raises(ValueError):
        _candle(close=float("nan"))


def test_candle_rejects_negative_volume() -> None:
    with pytest.raises(ValueError):
        _candle(volume=-1.0)


def _position(**overrides: object) -> Position:
    payload: dict[str, object] = {
        "symbol": "BTC/USD",
        "side": Side.LONG,
        "qty": 2.0,
        "entry_price": 100.0,
        "stop_price": 98.0,
        "opened_at": datetime(2024, 1, 1, tzinfo=UTC),
    }
    payload.update(overrides)
    return Position(**payload)  # type: ignore[arg-type]


def test_position_math() -> None:
    position = _position()
    assert position.risk_per_unit == 2.0
    assert position.risk_amount == 4.0
    assert position.gross_pnl(101.0) == pytest.approx(2.0)
    assert position.r_multiple(102.0) == pytest.approx(1.0)


def test_position_fees_are_separate_from_gross_pnl() -> None:
    position = _position(entry_fee=1.0)
    assert position.gross_pnl(101.0) == pytest.approx(2.0)
    assert position.unrealized_pnl(101.0) == pytest.approx(1.0)


def test_position_rejects_stop_on_wrong_side() -> None:
    with pytest.raises(ValueError):
        _position(stop_price=101.0)
    with pytest.raises(ValueError):
        _position(side=Side.SHORT, stop_price=99.0)


def test_position_rejects_take_profit_on_wrong_side() -> None:
    with pytest.raises(ValueError):
        _position(take_profit_price=99.5)


def test_position_stop_only_tightens() -> None:
    position = _position()
    position.move_stop(99.0)
    assert position.stop_price == 99.0
    with pytest.raises(ValueError):
        position.move_stop(98.5)
    assert position.widened_stop(98.5)


def test_position_marks_excursions() -> None:
    position = _position()
    position.mark(_candle(open=100.0, high=105.0, low=97.0, close=104.0))
    assert position.max_favorable_price == 105.0
    assert position.max_adverse_price == 97.0


def test_short_position_marks_inverted() -> None:
    position = _position(side=Side.SHORT, stop_price=102.0)
    position.mark(_candle(open=100.0, high=105.0, low=97.0, close=99.0))
    assert position.max_favorable_price == 97.0
    assert position.max_adverse_price == 105.0


def test_signal_requires_stop_for_entries() -> None:
    with pytest.raises(ValueError):
        Signal(action=Signal.enter_long(99.0).action, reason="missing stop")
    signal = Signal.enter_long(99.0, "ok", take_profit_price=103.0)
    assert signal.action.is_entry
    assert signal.stop_price == 99.0
    assert signal.take_profit_price == 103.0


def test_signal_hold_drops_levels() -> None:
    signal = Signal.hold("nothing")
    assert signal.stop_price is None
    assert signal.take_profit_price is None
    assert not signal.action.is_entry


def test_signal_positive_stop_only() -> None:
    with pytest.raises(ValueError):
        Signal.enter_short(-1.0, "bad")


def test_fill_validates_and_computes_notional() -> None:
    fill = Fill(
        order_id="x",
        symbol="BTC/USD",
        side=OrderSide.BUY,
        qty=2.0,
        price=100.0,
        fee=0.2,
        ts=datetime(2024, 1, 1, tzinfo=UTC),
    )
    assert fill.notional == 200.0
    with pytest.raises(ValueError):
        Fill(
            order_id="x",
            symbol="BTC/USD",
            side=OrderSide.BUY,
            qty=0.0,
            price=100.0,
            fee=0.0,
            ts=datetime(2024, 1, 1, tzinfo=UTC),
        )


def test_order_requires_price_for_limit() -> None:
    with pytest.raises(ValueError):
        Order(
            id="x",
            symbol="BTC/USD",
            side=OrderSide.BUY,
            qty=1.0,
            kind=__import__("mtmbot.models", fromlist=["OrderKind"]).OrderKind.LIMIT,
            ts=datetime(2024, 1, 1, tzinfo=UTC),
        )


def test_trade_rejects_reversed_timestamps() -> None:
    with pytest.raises(ValueError):
        Trade(
            symbol="BTC/USD",
            side=Side.LONG,
            qty=1.0,
            entry_price=100.0,
            exit_price=101.0,
            entry_ts=datetime(2024, 1, 2, tzinfo=UTC),
            exit_ts=datetime(2024, 1, 1, tzinfo=UTC),
            fees=0.1,
            pnl=0.9,
            r_multiple=0.9,
            exit_reason="test",
        )
