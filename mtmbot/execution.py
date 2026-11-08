"""Broker adapters and the portfolio ledger.

The portfolio is the single source of truth for cash, open positions and
closed trades. Fees are subtracted from cash exactly once, on the fill that
caused them.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol, runtime_checkable

from mtmbot.config import CostsConfig
from mtmbot.models import (
    Fill,
    Order,
    OrderSide,
    Position,
    Side,
    Trade,
    ensure_finite,
    ensure_positive,
    ensure_utc,
)

__all__ = ["Broker", "BrokerError", "PaperBroker", "Portfolio"]


class BrokerError(RuntimeError):
    """Raised when an order cannot be executed."""


@runtime_checkable
class Broker(Protocol):
    """Anything that can turn an :class:`Order` into a :class:`Fill`."""

    def submit(self, order: Order, reference_price: float) -> Fill:
        """Execute ``order`` and return the resulting fill."""
        ...


@dataclass(slots=True)
class Portfolio:
    """Cash, open positions and the trade ledger."""

    starting_cash: float
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)
    trades: list[Trade] = field(default_factory=list)
    fees_paid: float = 0.0
    realized_pnl: float = 0.0

    def __post_init__(self) -> None:
        self.starting_cash = ensure_positive(self.starting_cash, "starting_cash")
        self.cash = ensure_finite(self.cash, "cash")

    @classmethod
    def create(cls, starting_cash: float) -> Portfolio:
        """Fresh portfolio with all cash available."""
        cash = ensure_positive(starting_cash, "starting_cash")
        return cls(starting_cash=cash, cash=cash)

    # ------------------------------------------------------------------ value
    def equity(self, marks: Mapping[str, float]) -> float:
        """Cash plus unrealised PnL at the given marks."""
        total = self.cash
        for symbol, position in self.positions.items():
            if symbol not in marks:
                raise KeyError(f"missing mark price for open position {symbol}")
            total += position.gross_pnl(marks[symbol])
        return total

    def exposure(self, marks: Mapping[str, float]) -> float:
        """Sum of absolute notional currently held."""
        return sum(
            abs(position.qty * marks.get(symbol, position.entry_price))
            for symbol, position in self.positions.items()
        )

    def open_position_count(self) -> int:
        """Number of open positions."""
        return len(self.positions)

    # ------------------------------------------------------------- mutations
    def open_position(
        self,
        *,
        symbol: str,
        side: Side,
        qty: float,
        price: float,
        stop_price: float,
        take_profit_price: float | None,
        ts: datetime,
        fee: float,
        reason: str = "",
    ) -> Position:
        """Record a new position and pay its entry fee."""
        if symbol in self.positions:
            raise BrokerError(f"already holding {symbol}; close it before opening another")
        amount = ensure_positive(qty, "qty")
        entry = ensure_positive(price, "price")
        cost = ensure_finite(fee, "fee")
        if cost < 0:
            raise ValueError(f"fee must be >= 0, got {cost}")
        position = Position(
            symbol=symbol,
            side=side,
            qty=amount,
            entry_price=entry,
            stop_price=stop_price,
            opened_at=ensure_utc(ts, "ts"),
            take_profit_price=take_profit_price,
            entry_fee=cost,
            reason=reason,
        )
        self.cash -= cost
        self.fees_paid += cost
        self.positions[symbol] = position
        return position

    def close_position(
        self,
        *,
        symbol: str,
        price: float,
        ts: datetime,
        fee: float,
        reason: str = "",
    ) -> Trade:
        """Close ``symbol``, book the trade and settle cash."""
        position = self.positions.get(symbol)
        if position is None:
            raise BrokerError(f"no open position for {symbol}")
        exit_price = ensure_positive(price, "price")
        cost = ensure_finite(fee, "fee")
        if cost < 0:
            raise ValueError(f"fee must be >= 0, got {cost}")

        gross = position.gross_pnl(exit_price)
        net = gross - cost - position.entry_fee
        self.cash += gross - cost
        self.fees_paid += cost
        self.realized_pnl += net
        del self.positions[symbol]

        risk_per_unit = position.risk_per_unit
        direction = position.side.sign
        r_multiple = net / (risk_per_unit * position.qty) if risk_per_unit > 0 else 0.0
        mae_r = (
            (position.max_adverse_price - position.entry_price) * direction / risk_per_unit
            if risk_per_unit > 0
            else 0.0
        )
        mfe_r = (
            (position.max_favorable_price - position.entry_price) * direction / risk_per_unit
            if risk_per_unit > 0
            else 0.0
        )
        trade = Trade(
            symbol=symbol,
            side=position.side,
            qty=position.qty,
            entry_price=position.entry_price,
            exit_price=exit_price,
            entry_ts=position.opened_at,
            exit_ts=ensure_utc(ts, "ts"),
            fees=position.entry_fee + cost,
            pnl=net,
            r_multiple=r_multiple,
            exit_reason=reason,
            mae_r=mae_r,
            mfe_r=mfe_r,
        )
        self.trades.append(trade)
        return trade


@dataclass(frozen=True, slots=True)
class PaperBroker:
    """Simulated exchange: slippage against you, taker fee on every fill."""

    costs: CostsConfig
    _ids: itertools.count[int] = field(default_factory=lambda: itertools.count(1), repr=False)

    def submit(self, order: Order, reference_price: float) -> Fill:
        """Fill a market order at ``reference_price`` plus adverse slippage."""
        if not isinstance(order, Order):
            raise TypeError(f"order must be an Order, got {type(order).__name__}")
        reference = ensure_positive(reference_price, "reference_price")
        slip = self.costs.slippage_rate
        if order.side is OrderSide.BUY:
            price = reference * (1.0 + slip)
        else:
            price = reference * (1.0 - slip)
        if price <= 0:  # pragma: no cover - impossible with sane slippage
            raise BrokerError("slippage produced a non-positive price")
        fee = price * order.qty * self.costs.taker_rate
        return Fill(
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            qty=order.qty,
            price=price,
            fee=fee,
            ts=order.ts,
            reason=order.reason,
        )

    def next_order_id(self) -> str:
        """Deterministic order id, useful for logs and tests."""
        return f"paper-{next(self._ids)}"
