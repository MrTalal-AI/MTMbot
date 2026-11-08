"""The single trading code path shared by backtest, paper and live modes.

One bar is processed in two phases, exactly like reality:

``begin_bar(price)``  execute whatever the previous bar decided, at the open.
``end_bar(candle)``   run protective exits inside the bar, trail the stop,
                      mark the book, then read the closed bar and queue a
                      decision for the next bar.

Because the decision is always queued for the *next* bar, the backtest cannot
use information it would not have had.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from datetime import datetime

from mtmbot.config import CostsConfig
from mtmbot.execution import Broker, BrokerError, PaperBroker, Portfolio
from mtmbot.models import (
    Action,
    Candle,
    Fill,
    Order,
    OrderKind,
    OrderSide,
    Position,
    Side,
    Signal,
    ensure_positive,
)
from mtmbot.risk import RiskManager
from mtmbot.strategy.mtm import MTMStrategy, PreparedMTM

__all__ = ["SessionError", "TradingSession"]

_LOG = logging.getLogger("mtmbot.session")


class SessionError(RuntimeError):
    """Raised when the session cannot continue safely."""


@dataclass(slots=True)
class TradingSession:
    """Coordinates strategy, risk, broker and portfolio for one symbol."""

    symbol: str
    portfolio: Portfolio
    risk: RiskManager
    broker: Broker
    strategy: MTMStrategy
    pending: Signal | None = None
    last_price: float = 0.0
    last_equity: float = 0.0
    _orders: itertools.count[int] = field(
        default_factory=lambda: itertools.count(1), init=False, repr=False
    )

    def __post_init__(self) -> None:
        if "/" not in self.symbol:
            raise SessionError(f"symbol must look like BASE/QUOTE, got {self.symbol!r}")
        self.last_equity = self.portfolio.cash

    # ------------------------------------------------------------------ utils
    def _next_order_id(self) -> str:
        if isinstance(self.broker, PaperBroker):
            return self.broker.next_order_id()
        return f"bot-{next(self._orders)}"

    def _equity(self, price: float) -> float:
        return self.portfolio.equity({self.symbol: price})

    # -------------------------------------------------------------- execution
    def _execute(self, order: Order, reference_price: float) -> Fill | None:
        """Submit ``order`` and return its fill, or ``None`` on a broker error."""
        try:
            return self.broker.submit(order, reference_price)
        except BrokerError as exc:
            _LOG.error("order rejected by broker: %s (%s)", exc, order.reason)
        except (ValueError, TypeError) as exc:
            _LOG.error("order failed validation: %s (%s)", exc, order.reason)
        except Exception as exc:  # network/exchange failures must never kill the loop
            _LOG.exception("unexpected broker failure: %s", exc)
        return None

    def _close_position(self, position: Position, reference_price: float, ts: datetime,
                        reason: str) -> None:
        order = Order(
            id=self._next_order_id(),
            symbol=self.symbol,
            side=OrderSide.SELL if position.side is Side.LONG else OrderSide.BUY,
            qty=position.qty,
            kind=OrderKind.MARKET,
            reduce_only=True,
            ts=ts,
            reason=reason,
        )
        fill = self._execute(order, reference_price)
        if fill is None:
            return
        try:
            trade = self.portfolio.close_position(
                symbol=self.symbol,
                price=fill.price,
                ts=ts,
                fee=fill.fee,
                reason=reason,
            )
        except (BrokerError, ValueError) as exc:
            _LOG.error("could not book closing trade: %s", exc)
            return
        _LOG.info(
            "closed %s %s qty=%.8g @ %.8g pnl=%.2f r=%.2f (%s)",
            self.symbol,
            trade.side.value,
            trade.qty,
            trade.exit_price,
            trade.pnl,
            trade.r_multiple,
            reason,
        )

    def _open_position(self, signal: Signal, price: float, ts: datetime) -> Position | None:
        side = signal.action.side
        stop = signal.stop_price
        if side is None or stop is None:  # pragma: no cover - Signal validates this
            _LOG.error("entry signal without side/stop; ignored")
            return None
        if self.portfolio.open_position_count() >= self.risk.config.max_positions:
            return None
        if self.risk.halted:
            _LOG.warning("entry skipped: %s", self.risk.halt_reason)
            return None

        equity = self._equity(price)
        gate = self.risk.can_open(self.portfolio.open_position_count(), equity)
        if not gate:
            _LOG.info("entry skipped: %s", gate.reason)
            return None
        sizing = self.risk.size(equity, price, stop, side=side)
        if not sizing:
            _LOG.info("entry skipped: %s", sizing.reason)
            return None

        order = Order(
            id=self._next_order_id(),
            symbol=self.symbol,
            side=OrderSide.BUY if side is Side.LONG else OrderSide.SELL,
            qty=sizing.qty,
            kind=OrderKind.MARKET,
            ts=ts,
            reason=signal.reason,
        )
        fill = self._execute(order, price)
        if fill is None:
            return None

        # Slippage must never leave us without a valid protective stop.
        invalid = (side is Side.LONG and fill.price <= stop) or (
            side is Side.SHORT and fill.price >= stop
        )
        if invalid:
            _LOG.warning(
                "fill at %.8g invalidated the %.8g stop; flattening immediately",
                fill.price,
                stop,
            )
            # Book the round trip so the ledger matches reality: the order was
            # filled, so the fees were paid even though the setup was void.
            placeholder_stop = (
                fill.price * 0.99 if side is Side.LONG else fill.price * 1.01
            )
            try:
                temporary = self.portfolio.open_position(
                    symbol=self.symbol,
                    side=side,
                    qty=fill.qty,
                    price=fill.price,
                    stop_price=placeholder_stop,
                    take_profit_price=None,
                    ts=ts,
                    fee=fill.fee,
                    reason="invalidated entry",
                )
            except (BrokerError, ValueError) as exc:
                _LOG.error("could not book invalidated entry: %s", exc)
                return None
            self._close_position(temporary, fill.price, ts, "invalidated entry")
            return None

        try:
            position = self.portfolio.open_position(
                symbol=self.symbol,
                side=side,
                qty=fill.qty,
                price=fill.price,
                stop_price=stop,
                take_profit_price=signal.take_profit_price,
                ts=ts,
                fee=fill.fee,
                reason=signal.reason,
            )
        except (BrokerError, ValueError) as exc:
            _LOG.error("could not book entry: %s", exc)
            return None
        _LOG.info(
            "opened %s %s qty=%.8g @ %.8g stop=%.8g tp=%s (%s)",
            self.symbol,
            side.value,
            position.qty,
            position.entry_price,
            position.stop_price,
            f"{position.take_profit_price:.8g}" if position.take_profit_price else "none",
            signal.reason,
        )
        return position

    # ------------------------------------------------------------------ phases
    def begin_bar(self, price: float, ts: datetime) -> None:
        """Execute the decision queued by the previous bar."""
        reference = ensure_positive(price, "price")
        self.last_price = reference
        signal = self.pending
        self.pending = None
        if signal is None or signal.action is Action.HOLD:
            return

        position = self.portfolio.positions.get(self.symbol)
        if signal.action is Action.EXIT:
            if position is not None:
                self._close_position(position, reference, ts, signal.reason or "signal exit")
            return

        side = signal.action.side
        if side is None:  # pragma: no cover - covered by Signal validation
            return
        if position is not None:
            if position.side is side:
                return  # already on the right side, nothing to do
            self._close_position(position, reference, ts, "flip to " + side.value)
        self._open_position(signal, reference, ts)

    def end_bar(self, candle: Candle, prepared: PreparedMTM, index: int) -> Signal:
        """Handle the bar that just closed and queue the next decision."""
        if index < 0 or index >= len(prepared.candles):
            raise SessionError(f"bar index {index} outside the prepared series")
        if prepared.candles[index].ts != candle.ts:
            raise SessionError("prepared series is out of sync with the current candle")

        position = self.portfolio.positions.get(self.symbol)
        # A position opened *after* this bar closed (live mode) must not be
        # judged by the bar's intrabar extremes: they happened before entry.
        if position is not None and position.opened_at <= candle.ts:
            position.mark(candle)
            self._check_protective_exits(position, candle)
            still_open = self.portfolio.positions.get(self.symbol)
            if still_open is not None:
                prepared.manage_position(still_open, index)

        self.last_price = candle.close
        self.last_equity = self._equity(candle.close)
        self.risk.update(self.last_equity, candle.ts)

        signal = prepared.signal(index, self.portfolio.positions.get(self.symbol))
        self.pending = None if signal.action is Action.HOLD else signal
        return signal

    def _check_protective_exits(self, position: Position, candle: Candle) -> None:
        """Stop/take-profit inside the bar. When both are touched, stop wins.

        Assuming the worse outcome is the only honest way to backtest a bar
        where intrabar order is unknown.
        """
        stop = position.stop_price
        take_profit = position.take_profit_price

        if position.side is Side.LONG:
            if candle.open <= stop:
                self._close_position(position, candle.open, candle.ts, "stop gap")
                return
            if candle.low <= stop:
                self._close_position(position, stop, candle.ts, "stop loss")
                return
            if take_profit is not None and candle.high >= take_profit:
                reference = max(candle.open, take_profit)
                self._close_position(position, reference, candle.ts, "take profit")
                return
        else:
            if candle.open >= stop:
                self._close_position(position, candle.open, candle.ts, "stop gap")
                return
            if candle.high >= stop:
                self._close_position(position, stop, candle.ts, "stop loss")
                return
            if take_profit is not None and candle.low <= take_profit:
                reference = min(candle.open, take_profit)
                self._close_position(position, reference, candle.ts, "take profit")
                return

    # ------------------------------------------------------------------ helpers
    def mark_to_market(self, price: float, ts: datetime) -> float:
        """Update risk state without trading (used between bars in live mode)."""
        equity = self._equity(price)
        self.last_price = price
        self.last_equity = equity
        self.risk.update(equity, ts)
        return equity

    @property
    def halt_reason(self) -> str | None:
        """Why entries are blocked, if they are."""
        return self.risk.halt_reason


def costs_for(config: CostsConfig) -> PaperBroker:
    """Small factory used by the CLI."""
    return PaperBroker(costs=config)
