"""Core immutable-ish domain models shared by every part of the bot.

Every price, quantity and fee that enters MTMbot is validated here. Rejecting
bad data at the boundary (instead of deep inside the risk maths) is the main
reason the bot can run unattended: a corrupt candle or a NaN quantity raises
immediately, before any money is risked.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum

__all__ = [
    "Action",
    "Candle",
    "Fill",
    "Order",
    "OrderKind",
    "OrderSide",
    "Position",
    "Side",
    "Signal",
    "Trade",
    "ensure_finite",
    "ensure_positive",
]


def ensure_finite(value: float, name: str) -> float:
    """Return ``value`` as a float, raising if it is not a finite number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return number


def ensure_positive(value: float, name: str) -> float:
    """Return ``value`` as a strictly positive float."""
    number = ensure_finite(value, name)
    if number <= 0:
        raise ValueError(f"{name} must be > 0, got {value!r}")
    return number


def ensure_utc(ts: datetime, name: str = "timestamp") -> datetime:
    """Normalise a datetime to timezone-aware UTC."""
    if not isinstance(ts, datetime):
        raise TypeError(f"{name} must be a datetime, got {type(ts).__name__}")
    if ts.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware (UTC); naive datetimes hide bugs")
    return ts.astimezone(UTC)


class Side(StrEnum):
    """Direction of a position."""

    LONG = "long"
    SHORT = "short"

    @property
    def sign(self) -> int:
        """+1 for long, -1 for short."""
        return 1 if self is Side.LONG else -1

    @property
    def opposite(self) -> Side:
        """The other side."""
        return Side.SHORT if self is Side.LONG else Side.LONG


class OrderSide(StrEnum):
    """Direction of a single order (what you send to the exchange)."""

    BUY = "buy"
    SELL = "sell"


class OrderKind(StrEnum):
    """Order type."""

    MARKET = "market"
    LIMIT = "limit"


class Action(StrEnum):
    """What the strategy wants to do on the current bar."""

    HOLD = "hold"
    ENTER_LONG = "enter_long"
    ENTER_SHORT = "enter_short"
    EXIT = "exit"

    @property
    def is_entry(self) -> bool:
        """True for entry signals."""
        return self in (Action.ENTER_LONG, Action.ENTER_SHORT)

    @property
    def side(self) -> Side | None:
        """Position side implied by the action, if any."""
        if self is Action.ENTER_LONG:
            return Side.LONG
        if self is Action.ENTER_SHORT:
            return Side.SHORT
        return None


@dataclass(frozen=True, slots=True)
class Candle:
    """A single OHLCV bar. ``ts`` is the bar's open time in UTC."""

    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "ts", ensure_utc(self.ts, "candle.ts"))
        for name in ("open", "high", "low", "close"):
            object.__setattr__(self, name, ensure_positive(getattr(self, name), f"candle.{name}"))
        object.__setattr__(self, "volume", ensure_finite(self.volume, "candle.volume"))
        if self.volume < 0:
            raise ValueError(f"candle.volume must be >= 0, got {self.volume!r}")
        if self.high < self.low:
            raise ValueError(f"candle.high ({self.high}) < candle.low ({self.low})")
        if not (self.low <= self.open <= self.high):
            raise ValueError(f"candle.open ({self.open}) outside [{self.low}, {self.high}]")
        if not (self.low <= self.close <= self.high):
            raise ValueError(f"candle.close ({self.close}) outside [{self.low}, {self.high}]")

    @property
    def range(self) -> float:
        """High minus low."""
        return self.high - self.low

    @property
    def is_bullish(self) -> bool:
        """Close at or above open."""
        return self.close >= self.open


@dataclass(frozen=True, slots=True)
class Signal:
    """Strategy output for one closed bar. Executed on the *next* bar open."""

    action: Action
    reason: str = ""
    stop_price: float | None = None
    take_profit_price: float | None = None
    strength: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.action, Action):
            raise TypeError(f"signal.action must be an Action, got {type(self.action).__name__}")
        object.__setattr__(self, "strength", ensure_finite(self.strength, "signal.strength"))
        if self.action.is_entry:
            if self.stop_price is None:
                raise ValueError("entry signals must carry a stop_price")
            object.__setattr__(
                self, "stop_price", ensure_positive(self.stop_price, "signal.stop_price")
            )
            if self.take_profit_price is not None:
                object.__setattr__(
                    self,
                    "take_profit_price",
                    ensure_positive(self.take_profit_price, "signal.take_profit_price"),
                )
        else:
            object.__setattr__(self, "stop_price", None)
            object.__setattr__(self, "take_profit_price", None)

    @classmethod
    def hold(cls, reason: str = "") -> Signal:
        """Do nothing."""
        return cls(Action.HOLD, reason=reason)

    @classmethod
    def exit(cls, reason: str = "") -> Signal:
        """Close an open position."""
        return cls(Action.EXIT, reason=reason)

    @classmethod
    def enter_long(
        cls, stop_price: float, reason: str = "", take_profit_price: float | None = None,
        strength: float = 0.0,
    ) -> Signal:
        """Open (or flip to) a long position."""
        return cls(
            Action.ENTER_LONG, reason=reason, stop_price=stop_price,
            take_profit_price=take_profit_price, strength=strength,
        )

    @classmethod
    def enter_short(
        cls, stop_price: float, reason: str = "", take_profit_price: float | None = None,
        strength: float = 0.0,
    ) -> Signal:
        """Open (or flip to) a short position."""
        return cls(
            Action.ENTER_SHORT, reason=reason, stop_price=stop_price,
            take_profit_price=take_profit_price, strength=strength,
        )


@dataclass(frozen=True, slots=True)
class Fill:
    """An executed order."""

    order_id: str
    symbol: str
    side: OrderSide
    qty: float
    price: float
    fee: float
    ts: datetime
    reason: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.side, OrderSide):
            raise TypeError(f"fill.side must be an OrderSide, got {type(self.side).__name__}")
        object.__setattr__(self, "qty", ensure_positive(self.qty, "fill.qty"))
        object.__setattr__(self, "price", ensure_positive(self.price, "fill.price"))
        object.__setattr__(self, "fee", ensure_finite(self.fee, "fill.fee"))
        if self.fee < 0:
            raise ValueError(f"fill.fee must be >= 0, got {self.fee!r}")
        object.__setattr__(self, "ts", ensure_utc(self.ts, "fill.ts"))

    @property
    def notional(self) -> float:
        """qty * price."""
        return self.qty * self.price


@dataclass(frozen=True, slots=True)
class Order:
    """A request to trade, created by risk sizing and sent to a broker."""

    id: str
    symbol: str
    side: OrderSide
    qty: float
    kind: OrderKind = OrderKind.MARKET
    price: float | None = None
    reduce_only: bool = False
    ts: datetime = field(default_factory=lambda: datetime.now(UTC))
    reason: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.side, OrderSide):
            raise TypeError(f"order.side must be an OrderSide, got {type(self.side).__name__}")
        if not isinstance(self.kind, OrderKind):
            raise TypeError(f"order.kind must be an OrderKind, got {type(self.kind).__name__}")
        object.__setattr__(self, "qty", ensure_positive(self.qty, "order.qty"))
        object.__setattr__(self, "ts", ensure_utc(self.ts, "order.ts"))
        if self.kind is OrderKind.LIMIT:
            if self.price is None:
                raise ValueError("limit orders require a price")
            object.__setattr__(self, "price", ensure_positive(self.price, "order.price"))
        elif self.price is not None:
            object.__setattr__(self, "price", ensure_positive(self.price, "order.price"))


@dataclass(slots=True)
class Position:
    """An open position with its protective stop."""

    symbol: str
    side: Side
    qty: float
    entry_price: float
    stop_price: float
    opened_at: datetime
    take_profit_price: float | None = None
    initial_stop_price: float = 0.0
    entry_fee: float = 0.0
    reason: str = ""
    max_favorable_price: float = 0.0
    max_adverse_price: float = 0.0
    breakeven_done: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.side, Side):
            raise TypeError(f"position.side must be a Side, got {type(self.side).__name__}")
        self.qty = ensure_positive(self.qty, "position.qty")
        self.entry_price = ensure_positive(self.entry_price, "position.entry_price")
        self.stop_price = ensure_positive(self.stop_price, "position.stop_price")
        self.opened_at = ensure_utc(self.opened_at, "position.opened_at")
        self.entry_fee = ensure_finite(self.entry_fee, "position.entry_fee")
        if self.take_profit_price is not None:
            self.take_profit_price = ensure_positive(
                self.take_profit_price, "position.take_profit_price"
            )
        if self.side is Side.LONG:
            if self.stop_price >= self.entry_price:
                raise ValueError(
                    f"long stop ({self.stop_price}) must be below entry ({self.entry_price})"
                )
            if self.take_profit_price is not None and self.take_profit_price <= self.entry_price:
                raise ValueError(
                    f"long take profit ({self.take_profit_price}) must be above entry "
                    f"({self.entry_price})"
                )
        else:
            if self.stop_price <= self.entry_price:
                raise ValueError(
                    f"short stop ({self.stop_price}) must be above entry ({self.entry_price})"
                )
            if self.take_profit_price is not None and self.take_profit_price >= self.entry_price:
                raise ValueError(
                    f"short take profit ({self.take_profit_price}) must be below entry "
                    f"({self.entry_price})"
                )
        # The initial stop defines 1R forever; the live stop may later move to
        # break-even or trail, and that must not change what "1R" means.
        if self.initial_stop_price:
            self.initial_stop_price = ensure_positive(
                self.initial_stop_price, "position.initial_stop_price"
            )
        else:
            self.initial_stop_price = self.stop_price
        if self.side is Side.LONG and self.initial_stop_price > self.entry_price:
            raise ValueError("initial stop must not be above entry for a long")
        if self.side is Side.SHORT and self.initial_stop_price < self.entry_price:
            raise ValueError("initial stop must not be below entry for a short")

        # Track the best/worst price seen since entry for R-multiple stats.
        self.max_favorable_price = self.max_favorable_price or self.entry_price
        self.max_adverse_price = self.max_adverse_price or self.entry_price

    @property
    def notional(self) -> float:
        """Position size in quote currency at entry."""
        return self.qty * self.entry_price

    @property
    def risk_per_unit(self) -> float:
        """Distance between entry and the initial stop, in quote currency."""
        return abs(self.entry_price - self.initial_stop_price)

    @property
    def risk_amount(self) -> float:
        """Total quote currency at risk if the stop is hit."""
        return self.risk_per_unit * self.qty

    def gross_pnl(self, price: float) -> float:
        """Open PnL at ``price`` before fees.

        Fees are never subtracted here: they are already taken out of the
        portfolio cash when a fill happens, so counting them twice would
        understate equity.
        """
        mark = ensure_positive(price, "price")
        return (mark - self.entry_price) * self.side.sign * self.qty

    def unrealized_pnl(self, price: float) -> float:
        """Open PnL at ``price``, net of the entry fee (for display)."""
        return self.gross_pnl(price) - self.entry_fee

    def r_multiple(self, price: float) -> float:
        """Open profit in units of initial risk (0.0 if the risk is zero)."""
        if self.risk_per_unit <= 0:
            return 0.0
        return (price - self.entry_price) * self.side.sign / self.risk_per_unit

    def widened_stop(self, new_stop: float) -> bool:
        """True if ``new_stop`` would *increase* risk (never allowed)."""
        if self.side is Side.LONG:
            return new_stop < self.stop_price
        return new_stop > self.stop_price

    def move_stop(self, new_stop: float) -> None:
        """Tighten the stop. Widening is rejected: stops only ever move one way."""
        if self.widened_stop(new_stop):
            raise ValueError(
                f"refusing to widen stop from {self.stop_price} to {new_stop} ({self.side.value})"
            )
        self.stop_price = ensure_positive(new_stop, "new_stop")

    def mark(self, candle: Candle) -> None:
        """Update best/worst excursion with a bar's extremes."""
        if self.side is Side.LONG:
            self.max_favorable_price = max(self.max_favorable_price, candle.high)
            self.max_adverse_price = min(self.max_adverse_price, candle.low)
        else:
            self.max_favorable_price = min(self.max_favorable_price, candle.low)
            self.max_adverse_price = max(self.max_adverse_price, candle.high)


@dataclass(frozen=True, slots=True)
class Trade:
    """A closed round trip."""

    symbol: str
    side: Side
    qty: float
    entry_price: float
    exit_price: float
    entry_ts: datetime
    exit_ts: datetime
    fees: float
    pnl: float
    r_multiple: float
    exit_reason: str
    mae_r: float = 0.0
    mfe_r: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "entry_ts", ensure_utc(self.entry_ts, "trade.entry_ts"))
        object.__setattr__(self, "exit_ts", ensure_utc(self.exit_ts, "trade.exit_ts"))
        if self.exit_ts < self.entry_ts:
            raise ValueError("trade.exit_ts is before trade.entry_ts")

    @property
    def is_win(self) -> bool:
        """Profitable after fees."""
        return self.pnl > 0

    @property
    def return_pct(self) -> float:
        """Return relative to entry notional, in percent."""
        if self.entry_price <= 0 or self.qty <= 0:
            return 0.0
        return 100.0 * self.pnl / (self.entry_price * self.qty)

    @property
    def holding_seconds(self) -> float:
        """Seconds in the trade."""
        return (self.exit_ts - self.entry_ts).total_seconds()
