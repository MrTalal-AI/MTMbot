"""Real-money broker adapter built on ccxt (optional dependency).

This module is deliberately conservative:

* it refuses to start unless ``MTMBOT_LIVE_OK=1`` is set by the operator;
* it refuses to start without API credentials in the environment;
* every exchange error is wrapped in :class:`BrokerError`, so the session can
  log it and keep the loop alive instead of crashing mid-position;
* it is the only place in the codebase that can move real money.

Verify it against an exchange testnet before using real funds. It is not
covered by the test suite because that would require live credentials.
"""

from __future__ import annotations

import contextlib
import logging
import os
from datetime import UTC, datetime
from typing import Any

from mtmbot.config import ExchangeConfig
from mtmbot.execution import BrokerError
from mtmbot.models import Fill, Order, OrderSide, ensure_positive

__all__ = ["CcxtBroker"]

_LOG = logging.getLogger("mtmbot.ccxt")

LIVE_GATE = "MTMBOT_LIVE_OK"


def _require_live_gate() -> None:
    if os.environ.get(LIVE_GATE) != "1":
        raise BrokerError(
            f"live trading is disabled: set {LIVE_GATE}=1 in the environment only after "
            "you have reviewed the configuration and tested on a paper account"
        )


class CcxtBroker:
    """Market-order execution against any ccxt-supported exchange."""

    def __init__(self, config: ExchangeConfig, symbol: str) -> None:
        if not isinstance(config, ExchangeConfig):
            raise TypeError(f"config must be an ExchangeConfig, got {type(config).__name__}")
        _require_live_gate()
        api_key = os.environ.get(config.api_key_env)
        api_secret = os.environ.get(config.api_secret_env)
        if not api_key or not api_secret:
            raise BrokerError(
                f"missing credentials: set {config.api_key_env} and "
                f"{config.api_secret_env} in the environment"
            )
        try:
            import ccxt
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise BrokerError(
                "ccxt is not installed; run: pip install 'mtmbot[live]'"
            ) from exc

        exchange_class = getattr(ccxt, config.id, None)
        if exchange_class is None:  # pragma: no cover - depends on ccxt version
            raise BrokerError(f"ccxt has no exchange named {config.id!r}")
        try:
            client = exchange_class(
                {
                    "apiKey": api_key,
                    "secret": api_secret,
                    "enableRateLimit": True,
                    "options": {"defaultType": config.market_type},
                }
            )
            if config.sandbox:
                client.set_sandbox_mode(True)
            client.load_markets()
        except Exception as exc:  # pragma: no cover - network/exchange dependent
            raise BrokerError(f"could not connect to {config.id}: {exc}") from exc

        self._ccxt = ccxt
        self.client = client
        self.symbol = symbol
        self.exchange_id = config.id
        self.sandbox = config.sandbox
        _LOG.warning(
            "LIVE broker ready on %s (sandbox=%s) for %s",
            config.id,
            config.sandbox,
            symbol,
        )

    def submit(self, order: Order, reference_price: float) -> Fill:
        """Send a market order and normalise the exchange response."""
        ensure_positive(reference_price, "reference_price")
        side = "buy" if order.side is OrderSide.BUY else "sell"
        ccxt = self._ccxt
        try:
            raw: dict[str, Any] = self.client.create_order(
                order.symbol, "market", side, order.qty, None
            )
        except ccxt.NetworkError as exc:  # pragma: no cover - network dependent
            raise BrokerError(f"network error while placing {side} order: {exc}") from exc
        except Exception as exc:  # pragma: no cover - exchange dependent
            raise BrokerError(f"exchange rejected {side} order: {exc}") from exc

        filled = raw.get("filled") or raw.get("amount") or order.qty
        price = raw.get("average") or raw.get("price") or reference_price
        try:
            filled_qty = float(filled)
            fill_price = float(price)
        except (TypeError, ValueError) as exc:  # pragma: no cover - malformed response
            raise BrokerError(f"exchange returned an unusable fill: {raw!r}") from exc
        if filled_qty <= 0 or fill_price <= 0:  # pragma: no cover
            raise BrokerError(f"exchange reported an empty fill: {raw!r}")

        fee = 0.0
        fee_info = raw.get("fee")
        if isinstance(fee_info, dict):
            try:
                fee = abs(float(fee_info.get("cost") or 0.0))
            except (TypeError, ValueError):
                fee = 0.0
        if fee == 0.0 and isinstance(raw.get("fees"), list):
            for item in raw["fees"]:
                if isinstance(item, dict):
                    try:
                        fee += abs(float(item.get("cost") or 0.0))
                    except (TypeError, ValueError):
                        continue

        timestamp = raw.get("timestamp")
        moment = (
            datetime.fromtimestamp(float(timestamp) / 1000.0, tz=UTC)
            if isinstance(timestamp, (int, float))
            else order.ts
        )
        return Fill(
            order_id=str(raw.get("id") or order.id),
            symbol=order.symbol,
            side=order.side,
            qty=filled_qty,
            price=fill_price,
            fee=fee,
            ts=moment,
            reason=order.reason,
        )

    def close(self) -> None:  # pragma: no cover - trivial
        """Release the exchange session."""
        with contextlib.suppress(Exception):
            self.client.close()
