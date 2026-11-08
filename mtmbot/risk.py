"""Position sizing and hard risk limits.

Two rules matter more than any indicator:

1. Size from the distance to the stop, never from a "feeling" about the setup.
2. Stop trading when the account is bleeding (daily loss limit) or when the
   drawdown limit is hit. Hitting the drawdown limit latches until a human
   resets it, which is what you want after a strategy stops matching reality.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, date, datetime

from mtmbot.config import RiskConfig
from mtmbot.models import Side, ensure_finite, ensure_positive

__all__ = ["RiskDecision", "RiskManager"]


@dataclass(frozen=True, slots=True)
class RiskDecision:
    """Outcome of a pre-trade check."""

    allowed: bool
    reason: str = ""
    qty: float = 0.0
    risk_amount: float = 0.0
    notional: float = 0.0

    def __bool__(self) -> bool:
        return self.allowed


class RiskManager:
    """Stateful risk gate: sizing, daily loss limit, drawdown kill switch."""

    def __init__(self, config: RiskConfig, starting_equity: float) -> None:
        if not isinstance(config, RiskConfig):
            raise TypeError(f"config must be a RiskConfig, got {type(config).__name__}")
        equity = ensure_positive(starting_equity, "starting_equity")
        self.config = config
        self.peak_equity = equity
        self.day_start_equity = equity
        self.day: date | None = None
        self.halt_reason: str | None = None
        self._daily_halt = False

    # ------------------------------------------------------------------ state
    @property
    def halted(self) -> bool:
        """True when new entries must be refused."""
        return self.halt_reason is not None

    def update(self, equity: float, ts: datetime | None = None) -> None:
        """Feed current equity in; updates the peak and both kill switches."""
        value = ensure_finite(equity, "equity")
        if value <= 0:
            self.halt_reason = self.halt_reason or "equity reached zero"
            return
        moment = (ts or datetime.now(UTC)).astimezone(UTC)
        today = moment.date()
        if self.day != today:
            self.day = today
            self.day_start_equity = value
            if self._daily_halt:  # a new day clears the daily halt only
                self._daily_halt = False
                self.halt_reason = None
        self.peak_equity = max(self.peak_equity, value)

        drawdown_pct = 100.0 * (self.peak_equity - value) / self.peak_equity
        if drawdown_pct >= self.config.max_drawdown_pct:
            self._daily_halt = False
            self.halt_reason = (
                f"max drawdown limit hit: {drawdown_pct:.2f}% "
                f"(limit {self.config.max_drawdown_pct:.2f}%)"
            )
            return
        daily_pct = 100.0 * (self.day_start_equity - value) / self.day_start_equity
        if daily_pct >= self.config.daily_loss_limit_pct:
            self._daily_halt = True
            self.halt_reason = self._daily_reason(daily_pct)

    def _daily_reason(self, pct: float | None = None) -> str:
        value = self.config.daily_loss_limit_pct if pct is None else pct
        return (
            f"daily loss limit hit: {value:.2f}% "
            f"(limit {self.config.daily_loss_limit_pct:.2f}%)"
        )

    def reset_halt(self) -> None:
        """Manual reset, for an operator who has reviewed the situation."""
        self.halt_reason = None
        self.day_start_equity = self.peak_equity
        self._daily_halt = False

    def snapshot(self) -> dict[str, object]:
        """Serialisable state, written by the live state file."""
        return {
            "peak_equity": self.peak_equity,
            "day": self.day.isoformat() if self.day else None,
            "day_start_equity": self.day_start_equity,
            "halt_reason": self.halt_reason,
            "daily_halt": self._daily_halt,
        }

    def restore(self, data: dict[str, object]) -> None:
        """Restore a snapshot written by :meth:`snapshot`."""
        peak = data.get("peak_equity")
        if isinstance(peak, (int, float)) and float(peak) > 0:
            self.peak_equity = float(peak)
        day = data.get("day")
        if isinstance(day, str):
            try:
                self.day = date.fromisoformat(day)
            except ValueError:
                self.day = None
        day_start = data.get("day_start_equity")
        if isinstance(day_start, (int, float)) and float(day_start) > 0:
            self.day_start_equity = float(day_start)
        halt = data.get("halt_reason")
        self.halt_reason = halt if isinstance(halt, str) and halt else None
        self._daily_halt = bool(data.get("daily_halt"))

    # ---------------------------------------------------------------- sizing
    def check_stop(self, entry_price: float, stop_price: float, side: Side) -> str | None:
        """Return a rejection reason if the stop is unusable, else ``None``."""
        if not isinstance(side, Side):
            raise TypeError(f"side must be a Side, got {type(side).__name__}")
        entry = ensure_positive(entry_price, "entry_price")
        stop = ensure_positive(stop_price, "stop_price")
        if side is Side.LONG and stop >= entry:
            return f"long stop {stop:.8g} is not below entry {entry:.8g}"
        if side is Side.SHORT and stop <= entry:
            return f"short stop {stop:.8g} is not above entry {entry:.8g}"
        distance_pct = 100.0 * abs(entry - stop) / entry
        if distance_pct > self.config.max_stop_distance_pct:
            return (
                f"stop is {distance_pct:.2f}% away, wider than the "
                f"{self.config.max_stop_distance_pct:.2f}% limit"
            )
        if distance_pct <= 0.0:  # pragma: no cover - implied by the checks above
            return "stop distance is zero"
        return None

    def size(
        self,
        equity: float,
        entry_price: float,
        stop_price: float,
        *,
        side: Side = Side.LONG,
        qty_step: float = 1e-8,
    ) -> RiskDecision:
        """Position size that risks exactly ``risk_per_trade_pct`` of equity."""
        account = ensure_finite(equity, "equity")
        if account <= 0:
            return RiskDecision(False, "equity is not positive")
        rejection = self.check_stop(entry_price, stop_price, side)
        if rejection is not None:
            return RiskDecision(False, rejection)
        if self.halted:
            return RiskDecision(False, self.halt_reason or "trading halted")
        if qty_step <= 0:
            raise ValueError(f"qty_step must be positive, got {qty_step!r}")

        risk_amount = account * self.config.risk_per_trade_pct / 100.0
        per_unit = abs(entry_price - stop_price)
        qty = risk_amount / per_unit

        max_notional = account * self.config.max_notional_pct / 100.0
        qty = min(qty, max_notional / entry_price)

        if qty_step > 0:
            steps = math.floor(qty / qty_step + 1e-12)
            qty = steps * qty_step
        if qty <= 0:
            return RiskDecision(False, "position size rounds to zero")

        notional = qty * entry_price
        if notional < self.config.min_notional:
            return RiskDecision(
                False,
                f"notional {notional:.2f} is below the minimum {self.config.min_notional:.2f}",
            )
        return RiskDecision(
            True,
            "",
            qty=qty,
            risk_amount=qty * per_unit,
            notional=notional,
        )

    def can_open(self, open_positions: int, equity: float) -> RiskDecision:
        """Check the portfolio-level limits before opening anything."""
        if isinstance(open_positions, bool) or not isinstance(open_positions, int):
            raise TypeError("open_positions must be an int")
        if open_positions < 0:
            raise ValueError(f"open_positions must be >= 0, got {open_positions}")
        if self.halted:
            return RiskDecision(False, self.halt_reason or "trading halted")
        if open_positions >= self.config.max_positions:
            return RiskDecision(
                False,
                f"already at the maximum of {self.config.max_positions} open position(s)",
            )
        if equity <= 0:
            return RiskDecision(False, "equity is not positive")
        return RiskDecision(True)
