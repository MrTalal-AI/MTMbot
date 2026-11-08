from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import pytest

from mtmbot.config import StrategyConfig
from mtmbot.models import Action, Position, Side
from mtmbot.strategy import MTMStrategy


def _config(**overrides: object) -> StrategyConfig:
    base = StrategyConfig(
        fast_ema=10,
        slow_ema=30,
        rsi_period=14,
        atr_period=14,
        htf_factor=4,
        atr_stop_mult=2.0,
        reward_risk=2.0,
        rsi_long_min=52.0,
        rsi_short_max=48.0,
    )
    return replace(base, **overrides)  # type: ignore[arg-type]


def _signals(prepared, action: Action) -> list[int]:
    return [
        index
        for index in range(len(prepared.candles))
        if prepared.signal(index, None).action is action
    ]


def test_uptrend_produces_valid_long_entries(uptrend_candles) -> None:
    prepared = MTMStrategy(_config(), 3600).prepare_from_base(uptrend_candles)
    entries = _signals(prepared, Action.ENTER_LONG)
    assert entries, "a long uptrend must eventually trigger a long entry"
    for index in entries:
        signal = prepared.signal(index, None)
        close = uptrend_candles[index].close
        assert signal.stop_price is not None and signal.stop_price < close
        assert signal.take_profit_price is not None and signal.take_profit_price > close
        assert prepared.bias_at(index) == 1
        assert prepared.rsi_values[index] is not None
        assert prepared.rsi_values[index] >= 52.0


def test_downtrend_produces_valid_short_entries(candle_factory) -> None:
    closes = []
    price = 500.0
    for index in range(700):
        price *= 0.996 if not 300 <= index < 306 else 1.003
        closes.append(price)
    candles = candle_factory(closes)
    prepared = MTMStrategy(_config(), 3600).prepare_from_base(candles)
    entries = _signals(prepared, Action.ENTER_SHORT)
    assert entries
    for index in entries:
        signal = prepared.signal(index, None)
        close = candles[index].close
        assert signal.stop_price is not None and signal.stop_price > close
        assert signal.take_profit_price is not None and signal.take_profit_price < close
        assert prepared.bias_at(index) == -1


def test_shorts_can_be_disabled(candle_factory) -> None:
    closes = [500.0 * 0.996**index for index in range(700)]
    candles = candle_factory(closes)
    prepared = MTMStrategy(_config(allow_shorts=False), 3600).prepare_from_base(candles)
    assert _signals(prepared, Action.ENTER_SHORT) == []


def test_flat_market_never_signals(flat_candles) -> None:
    prepared = MTMStrategy(_config(), 3600).prepare_from_base(flat_candles)
    assert _signals(prepared, Action.ENTER_LONG) == []
    assert _signals(prepared, Action.ENTER_SHORT) == []


def test_warmup_blocks_early_signals(uptrend_candles) -> None:
    strategy = MTMStrategy(_config(), 3600)
    prepared = strategy.prepare_from_base(uptrend_candles)
    assert not prepared.ready[0]
    assert prepared.signal(0, None).action is Action.HOLD
    assert strategy.warmup_bars > 0


def test_bias_flip_exits_position(uptrend_candles) -> None:
    prepared = MTMStrategy(_config(), 3600).prepare_from_base(uptrend_candles)
    position = Position(
        symbol="BTC/USD",
        side=Side.SHORT,
        qty=1.0,
        entry_price=200.0,
        stop_price=210.0,
        opened_at=uptrend_candles[-1].ts,
    )
    signal = prepared.signal(prepared.last_index, position)
    assert signal.action is Action.EXIT


def test_manage_position_moves_stop_to_breakeven_and_beyond(candle_factory) -> None:
    closes = [100.0 * 1.003**index for index in range(400)]
    candles = candle_factory(closes)
    prepared = MTMStrategy(_config(), 3600).prepare_from_base(candles)

    position = Position(
        symbol="BTC/USD",
        side=Side.LONG,
        qty=1.0,
        entry_price=100.0,
        stop_price=98.0,
        opened_at=candles[0].ts,
    )
    previous = position.stop_price
    for index in range(len(candles)):
        prepared.manage_position(position, index)
        assert position.stop_price >= previous, "stops may only tighten"
        assert position.stop_price < closes[index]
        previous = position.stop_price

    assert position.stop_price >= position.entry_price
    assert position.breakeven_done


def test_manage_position_ignores_warmup_indices(uptrend_candles) -> None:
    prepared = MTMStrategy(_config(), 3600).prepare_from_base(uptrend_candles)
    position = Position(
        symbol="BTC/USD",
        side=Side.LONG,
        qty=1.0,
        entry_price=100.0,
        stop_price=98.0,
        opened_at=uptrend_candles[0].ts,
    )
    prepared.manage_position(position, 0)
    assert position.stop_price == 98.0


def test_prepare_rejects_empty_series() -> None:
    with pytest.raises(ValueError):
        MTMStrategy(_config(), 3600).prepare([], [])


def test_bias_stays_neutral_until_the_htf_bar_closes(uptrend_candles) -> None:
    prepared = MTMStrategy(_config(), 3600).prepare_from_base(uptrend_candles)
    # htf_factor=4 with 1h bars: the first 4h bar closes after bar index 3.
    assert [prepared.bias_at(index) for index in range(4)] == [0, 0, 0, 0]
    # Still neutral: the higher-timeframe EMAs themselves need warm-up.
    assert prepared.bias_at(4) == 0
    assert prepared.bias_at(200) == 1


def test_strategy_rejects_bad_step() -> None:
    with pytest.raises(ValueError):
        MTMStrategy(_config(), 0)


def test_prepared_candles_match_input(uptrend_candles) -> None:
    prepared = MTMStrategy(_config(), 3600).prepare_from_base(uptrend_candles)
    assert len(prepared) == len(uptrend_candles)
    assert prepared.last_index == len(uptrend_candles) - 1
    assert isinstance(prepared.candles[0].ts, datetime)
    assert prepared.candles[0].ts.tzinfo is UTC
