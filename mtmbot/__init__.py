"""MTMbot - Multi-Timeframe Momentum trading bot.

The same strategy, risk manager and execution code path is used for
backtesting, paper trading and live trading, so a backtest cannot silently
diverge from what the bot actually does.
"""

from __future__ import annotations

from mtmbot.models import Action, Candle, Fill, Order, OrderSide, Position, Side, Signal, Trade

__version__ = "1.0.0"

__all__ = [
    "Action",
    "Candle",
    "Fill",
    "Order",
    "OrderSide",
    "Position",
    "Side",
    "Signal",
    "Trade",
    "__version__",
]
