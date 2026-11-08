"""Crash-safe persistence for the live/paper engine.

A bot that forgets its position after a restart will happily open a second one,
so state is written atomically after every processed bar and *validated* on
load: a corrupt state file stops the bot instead of being silently ignored.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from mtmbot.models import Action, Position, Side, Signal, Trade
from mtmbot.session import TradingSession

__all__ = ["STATE_VERSION", "StateError", "restore_state", "save_state"]

STATE_VERSION = 1


class StateError(RuntimeError):
    """Raised when the state file exists but cannot be trusted."""


def _position_to_dict(position: Position) -> dict[str, Any]:
    return {
        "symbol": position.symbol,
        "side": position.side.value,
        "qty": position.qty,
        "entry_price": position.entry_price,
        "stop_price": position.stop_price,
        "initial_stop_price": position.initial_stop_price,
        "take_profit_price": position.take_profit_price,
        "opened_at": position.opened_at.isoformat(),
        "entry_fee": position.entry_fee,
        "reason": position.reason,
        "max_favorable_price": position.max_favorable_price,
        "max_adverse_price": position.max_adverse_price,
        "breakeven_done": position.breakeven_done,
    }


def _position_from_dict(data: Any) -> Position:
    if not isinstance(data, dict):
        raise StateError("position entry is not an object")
    try:
        take_profit = data.get("take_profit_price")
        return Position(
            symbol=str(data["symbol"]),
            side=Side(str(data["side"])),
            qty=float(data["qty"]),
            entry_price=float(data["entry_price"]),
            stop_price=float(data["stop_price"]),
            opened_at=datetime.fromisoformat(str(data["opened_at"])),
            take_profit_price=None if take_profit is None else float(take_profit),
            initial_stop_price=float(data.get("initial_stop_price") or data["stop_price"]),
            entry_fee=float(data.get("entry_fee", 0.0)),
            reason=str(data.get("reason", "")),
            max_favorable_price=float(data.get("max_favorable_price", 0.0)),
            max_adverse_price=float(data.get("max_adverse_price", 0.0)),
            breakeven_done=bool(data.get("breakeven_done", False)),
        )
    except (KeyError, ValueError, TypeError) as exc:
        raise StateError(f"invalid position in state file: {exc}") from exc


def _trade_to_dict(trade: Trade) -> dict[str, Any]:
    return {
        "symbol": trade.symbol,
        "side": trade.side.value,
        "qty": trade.qty,
        "entry_price": trade.entry_price,
        "exit_price": trade.exit_price,
        "entry_ts": trade.entry_ts.isoformat(),
        "exit_ts": trade.exit_ts.isoformat(),
        "fees": trade.fees,
        "pnl": trade.pnl,
        "r_multiple": trade.r_multiple,
        "exit_reason": trade.exit_reason,
        "mae_r": trade.mae_r,
        "mfe_r": trade.mfe_r,
    }


def _trade_from_dict(data: Any) -> Trade:
    if not isinstance(data, dict):
        raise StateError("trade entry is not an object")
    try:
        return Trade(
            symbol=str(data["symbol"]),
            side=Side(str(data["side"])),
            qty=float(data["qty"]),
            entry_price=float(data["entry_price"]),
            exit_price=float(data["exit_price"]),
            entry_ts=datetime.fromisoformat(str(data["entry_ts"])),
            exit_ts=datetime.fromisoformat(str(data["exit_ts"])),
            fees=float(data["fees"]),
            pnl=float(data["pnl"]),
            r_multiple=float(data["r_multiple"]),
            exit_reason=str(data.get("exit_reason", "")),
            mae_r=float(data.get("mae_r", 0.0)),
            mfe_r=float(data.get("mfe_r", 0.0)),
        )
    except (KeyError, ValueError, TypeError) as exc:
        raise StateError(f"invalid trade in state file: {exc}") from exc


def _signal_to_dict(signal: Signal | None) -> dict[str, Any] | None:
    if signal is None:
        return None
    return {
        "action": signal.action.value,
        "reason": signal.reason,
        "stop_price": signal.stop_price,
        "take_profit_price": signal.take_profit_price,
        "strength": signal.strength,
    }


def _signal_from_dict(data: Any) -> Signal | None:
    if data is None:
        return None
    if not isinstance(data, dict):
        raise StateError("pending signal is not an object")
    try:
        return Signal(
            action=Action(str(data["action"])),
            reason=str(data.get("reason", "")),
            stop_price=data.get("stop_price"),
            take_profit_price=data.get("take_profit_price"),
            strength=float(data.get("strength", 0.0)),
        )
    except (KeyError, ValueError, TypeError) as exc:
        raise StateError(f"invalid pending signal in state file: {exc}") from exc


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=".state-", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        Path(tmp_name).replace(path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def save_state(path: str | Path, session: TradingSession, last_bar_ts: datetime | None) -> Path:
    """Write the full session state to ``path`` atomically."""
    target = Path(path)
    payload = {
        "version": STATE_VERSION,
        "saved_at": datetime.now().astimezone().isoformat(),
        "symbol": session.symbol,
        "cash": session.portfolio.cash,
        "starting_cash": session.portfolio.starting_cash,
        "fees_paid": session.portfolio.fees_paid,
        "realized_pnl": session.portfolio.realized_pnl,
        "last_price": session.last_price,
        "last_equity": session.last_equity,
        "positions": [_position_to_dict(p) for p in session.portfolio.positions.values()],
        "trades": [_trade_to_dict(t) for t in session.portfolio.trades],
        "pending": _signal_to_dict(session.pending),
        "risk": session.risk.snapshot(),
        "last_bar_ts": last_bar_ts.isoformat() if last_bar_ts else None,
    }
    _atomic_write(target, json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return target


def restore_state(path: str | Path, session: TradingSession) -> datetime | None:
    """Load state into ``session``. Returns the last processed bar time."""
    target = Path(path)
    if not target.exists():
        return None
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StateError(f"cannot read state file {target}: {exc}") from exc
    if not isinstance(payload, dict):
        raise StateError(f"state file {target} is not an object")
    version = payload.get("version")
    if version != STATE_VERSION:
        raise StateError(
            f"state file version {version!r} is not supported (expected {STATE_VERSION})"
        )
    if payload.get("symbol") != session.symbol:
        raise StateError(
            f"state file is for {payload.get('symbol')!r}, but the bot trades "
            f"{session.symbol!r}"
        )

    portfolio = session.portfolio
    try:
        portfolio.cash = float(payload["cash"])
        portfolio.starting_cash = float(payload.get("starting_cash", portfolio.starting_cash))
        portfolio.fees_paid = float(payload.get("fees_paid", 0.0))
        portfolio.realized_pnl = float(payload.get("realized_pnl", 0.0))
    except (KeyError, ValueError, TypeError) as exc:
        raise StateError(f"invalid portfolio in state file: {exc}") from exc

    positions = payload.get("positions", [])
    if not isinstance(positions, list):
        raise StateError("positions must be a list")
    portfolio.positions = {position.symbol: position for position in map(_position_from_dict,
                                                                     positions)}
    trades = payload.get("trades", [])
    if not isinstance(trades, list):
        raise StateError("trades must be a list")
    portfolio.trades = [_trade_from_dict(item) for item in trades]
    session.pending = _signal_from_dict(payload.get("pending"))
    session.last_price = float(payload.get("last_price", 0.0) or 0.0)
    session.last_equity = float(payload.get("last_equity", portfolio.cash) or portfolio.cash)

    risk_state = payload.get("risk")
    if isinstance(risk_state, dict):
        session.risk.restore(risk_state)

    last_bar = payload.get("last_bar_ts")
    if not last_bar:
        return None
    try:
        return datetime.fromisoformat(str(last_bar))
    except ValueError as exc:
        raise StateError(f"invalid last_bar_ts in state file: {last_bar!r}") from exc
