"""Multi-Timeframe Momentum (MTM) strategy.

Logic, in one sentence: trade only in the direction of the higher-timeframe
trend, enter on a momentum cross-back in that direction, and risk a fixed
fraction of equity between entry and an ATR-based stop.

Anti look-ahead rules baked into this module:

* Signals are computed on a bar's *close* and executed on the next bar's open.
* The higher-timeframe bias only ever uses bars that have fully closed before
  the current bar opened (see ``map_htf_indices``).
* Every indicator value is causal (see ``mtmbot.indicators``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from mtmbot.config import StrategyConfig
from mtmbot.data.resample import map_htf_indices, resample
from mtmbot.indicators import atr, crossover, crossunder, ema, rsi
from mtmbot.models import Action, Candle, Position, Side, Signal

__all__ = ["MTMStrategy", "PreparedMTM"]


@dataclass(frozen=True, slots=True)
class PreparedMTM:
    """Pre-computed, immutable indicator arrays for a candle series."""

    config: StrategyConfig
    candles: tuple[Candle, ...]
    fast: tuple[float | None, ...]
    slow: tuple[float | None, ...]
    rsi_values: tuple[float | None, ...]
    atr_values: tuple[float | None, ...]
    htf_bias: tuple[int, ...]
    ready: tuple[bool, ...]
    htf_bars: int

    def __len__(self) -> int:
        return len(self.candles)

    @property
    def last_index(self) -> int:
        """Index of the most recent bar."""
        return len(self.candles) - 1

    def bias_at(self, index: int) -> int:
        """+1 bullish, -1 bearish, 0 undecided at ``index``."""
        if index < 0 or index >= len(self.htf_bias):
            return 0
        return self.htf_bias[index]

    def is_ready(self, index: int) -> bool:
        """True when every input needed for a decision exists at ``index``."""
        return 0 <= index < len(self.ready) and self.ready[index]

    def signal(self, index: int, position: Position | None) -> Signal:
        """Decision for the bar at ``index`` (already closed)."""
        if not self.is_ready(index):
            return Signal.hold("warmup")
        config = self.config
        candle = self.candles[index]
        close = candle.close
        atr_value = self.atr_values[index]
        bias = self.htf_bias[index]
        if atr_value is None or atr_value <= 0:  # pragma: no cover - guarded by ready
            return Signal.hold("no atr")

        if position is not None:
            if config.exit_on_bias_flip and bias != 0 and bias == -position.side.sign:
                return Signal.exit("higher-timeframe bias flipped against position")
            return Signal.hold("managing open position")

        if bias == 0:
            return Signal.hold("no higher-timeframe trend")

        slow = self.slow[index]
        rsi_now = self.rsi_values[index]
        if slow is None or rsi_now is None:  # pragma: no cover - guarded by ready
            return Signal.hold("warmup")

        stop_distance = config.atr_stop_mult * atr_value
        if stop_distance <= 0:  # pragma: no cover - guarded above
            return Signal.hold("degenerate volatility")

        if bias > 0:
            if close <= slow:
                return Signal.hold("price below slow EMA")
            if rsi_now < config.rsi_long_min:
                return Signal.hold("momentum too weak")
            if not crossover(self._close_series(), self.fast, index):
                return Signal.hold("no momentum trigger")
            stop = close - stop_distance
            if stop <= 0:
                return Signal.hold("stop would be negative")
            take_profit = close + config.reward_risk * stop_distance
            strength = min(1.0, max(0.0, (rsi_now - 50.0) / 50.0))
            return Signal.enter_long(
                stop_price=stop,
                take_profit_price=take_profit,
                reason=(
                    f"HTF bullish, close crossed EMA{config.fast_ema}, "
                    f"RSI {rsi_now:.1f} >= {config.rsi_long_min:.0f}"
                ),
                strength=strength,
            )

        if not config.allow_shorts:
            return Signal.hold("shorts disabled")
        if close >= slow:
            return Signal.hold("price above slow EMA")
        if rsi_now > config.rsi_short_max:
            return Signal.hold("momentum too weak")
        if not crossunder(self._close_series(), self.fast, index):
            return Signal.hold("no momentum trigger")
        stop = close + stop_distance
        take_profit = close - config.reward_risk * stop_distance
        if take_profit <= 0:
            return Signal.hold("take profit would be negative")
        strength = min(1.0, max(0.0, (50.0 - rsi_now) / 50.0))
        return Signal.enter_short(
            stop_price=stop,
            take_profit_price=take_profit,
            reason=(
                f"HTF bearish, close crossed below EMA{config.fast_ema}, "
                f"RSI {rsi_now:.1f} <= {config.rsi_short_max:.0f}"
            ),
            strength=strength,
        )

    def _close_series(self) -> tuple[float | None, ...]:
        return tuple(candle.close for candle in self.candles)

    def manage_position(self, position: Position, index: int) -> None:
        """Break-even and trailing logic, applied after the bar's exits ran.

        The stop can only ever tighten, so a bug here can reduce profit but
        cannot silently increase risk.
        """
        if not 0 <= index < len(self.candles):
            return
        atr_value = self.atr_values[index]
        if atr_value is None or atr_value <= 0:
            return
        config = self.config
        price = self.candles[index].close
        r_now = position.r_multiple(price)

        if not position.breakeven_done and config.breakeven_at_r > 0 and r_now >= config.breakeven_at_r:
            if not position.widened_stop(position.entry_price):
                position.move_stop(position.entry_price)
            position.breakeven_done = True

        if config.trail_at_r > 0 and r_now >= config.trail_at_r:
            if position.side is Side.LONG:
                candidate = price - config.trail_atr_mult * atr_value
            else:
                candidate = price + config.trail_atr_mult * atr_value
            if candidate > 0 and not position.widened_stop(candidate):
                position.move_stop(candidate)


class MTMStrategy:
    """Builds :class:`PreparedMTM` for a candle series."""

    name = "mtm"

    def __init__(self, config: StrategyConfig, step_seconds: int) -> None:
        if step_seconds <= 0:
            raise ValueError(f"step_seconds must be positive, got {step_seconds!r}")
        if not isinstance(config, StrategyConfig):
            raise TypeError(f"config must be a StrategyConfig, got {type(config).__name__}")
        self.config = config
        self.step_seconds = step_seconds
        self.htf_seconds = step_seconds * config.htf_factor

    @property
    def warmup_bars(self) -> int:
        """Minimum number of base bars needed before any signal can fire."""
        base_need = max(
            self.config.slow_ema,
            self.config.rsi_period + 1,
            self.config.atr_period + 1,
        )
        htf_need = max(self.config.slow_ema, self.config.fast_ema) * self.config.htf_factor
        return base_need + htf_need

    def prepare(self, candles: Sequence[Candle], htf_candles: Sequence[Candle]) -> PreparedMTM:
        """Compute every array the strategy needs, in one pass."""
        if not candles:
            raise ValueError("cannot prepare a strategy on an empty candle series")
        closes = [candle.close for candle in candles]
        htf_closes = [candle.close for candle in htf_candles]

        fast = ema(closes, self.config.fast_ema)
        slow = ema(closes, self.config.slow_ema)
        rsi_values = rsi(closes, self.config.rsi_period)
        atr_values = atr(list(candles), self.config.atr_period)

        htf_fast = ema(htf_closes, self.config.fast_ema)
        htf_slow = ema(htf_closes, self.config.slow_ema)
        htf_bias: list[int] = []
        for index in range(len(htf_candles)):
            f, s = htf_fast[index], htf_slow[index]
            if f is None or s is None:
                htf_bias.append(0)
            elif f > s and htf_closes[index] > s:
                htf_bias.append(1)
            elif f < s and htf_closes[index] < s:
                htf_bias.append(-1)
            else:
                htf_bias.append(0)

        mapping = map_htf_indices(candles, htf_candles, self.htf_seconds)
        bias: list[int] = []
        ready: list[bool] = []
        for index in range(len(candles)):
            htf_index = mapping[index]
            if htf_index < 0:
                bias.append(0)
                ready.append(False)
                continue
            bias.append(htf_bias[htf_index])
            ready.append(
                fast[index] is not None
                and slow[index] is not None
                and rsi_values[index] is not None
                and atr_values[index] is not None
                and htf_fast[htf_index] is not None
                and htf_slow[htf_index] is not None
            )

        return PreparedMTM(
            config=self.config,
            candles=tuple(candles),
            fast=tuple(fast),
            slow=tuple(slow),
            rsi_values=tuple(rsi_values),
            atr_values=tuple(atr_values),
            htf_bias=tuple(bias),
            ready=tuple(ready),
            htf_bars=len(htf_candles),
        )

    def prepare_from_base(self, candles: Sequence[Candle]) -> PreparedMTM:
        """Convenience: resample internally, then prepare."""
        htf = resample(candles, self.step_seconds, self.config.htf_factor)
        return self.prepare(candles, htf)


def is_entry_signal(signal: Signal) -> bool:
    """Small helper kept for readability at call sites."""
    return signal.action.is_entry and signal.action is not Action.HOLD
