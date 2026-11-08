from __future__ import annotations

from datetime import UTC, datetime

import pytest

from mtmbot.config import CostsConfig
from mtmbot.execution import BrokerError, PaperBroker, Portfolio
from mtmbot.models import Order, OrderSide, Side

NOW = datetime(2024, 1, 1, tzinfo=UTC)


def _order(side: OrderSide, qty: float = 1.0) -> Order:
    return Order(id="o1", symbol="BTC/USD", side=side, qty=qty, ts=NOW)


def test_paper_broker_buy_pays_slippage_and_fee() -> None:
    broker = PaperBroker(costs=CostsConfig(taker_bps=10.0, slippage_bps=50.0))
    fill = broker.submit(_order(OrderSide.BUY, 2.0), 100.0)
    assert fill.price == pytest.approx(100.5)
    assert fill.fee == pytest.approx(100.5 * 2.0 * 0.001)
    assert fill.notional == pytest.approx(201.0)


def test_paper_broker_sell_receives_worse_price() -> None:
    broker = PaperBroker(costs=CostsConfig(taker_bps=0.0, slippage_bps=50.0))
    fill = broker.submit(_order(OrderSide.SELL, 1.0), 100.0)
    assert fill.price == pytest.approx(99.5)
    assert fill.fee == 0.0


def test_paper_broker_order_ids_are_unique() -> None:
    broker = PaperBroker(costs=CostsConfig())
    assert broker.next_order_id() != broker.next_order_id()


def test_portfolio_equity_includes_unrealised_pnl() -> None:
    portfolio = Portfolio.create(10_000.0)
    portfolio.open_position(
        symbol="BTC/USD",
        side=Side.LONG,
        qty=1.0,
        price=100.0,
        stop_price=95.0,
        take_profit_price=110.0,
        ts=NOW,
        fee=10.0,
    )
    marks = {"BTC/USD": 105.0}
    assert portfolio.cash == pytest.approx(9_990.0)
    # cash 9_990 plus 5.0 of open profit, entry fee already paid out of cash
    assert portfolio.equity(marks) == pytest.approx(9_995.0)
    assert portfolio.exposure(marks) == pytest.approx(105.0)


def test_portfolio_close_books_net_trade() -> None:
    portfolio = Portfolio.create(10_000.0)
    portfolio.open_position(
        symbol="BTC/USD",
        side=Side.LONG,
        qty=2.0,
        price=100.0,
        stop_price=98.0,
        take_profit_price=None,
        ts=NOW,
        fee=2.0,
    )
    trade = portfolio.close_position(
        symbol="BTC/USD", price=104.0, ts=NOW, fee=2.0, reason="take profit"
    )
    # gross 8, entry fee 2, exit fee 2
    assert trade.pnl == pytest.approx(4.0)
    assert trade.r_multiple == pytest.approx(1.0)
    assert trade.fees == pytest.approx(4.0)
    assert portfolio.cash == pytest.approx(10_004.0)
    assert portfolio.realized_pnl == pytest.approx(4.0)
    assert not portfolio.positions
    assert len(portfolio.trades) == 1


def test_portfolio_short_pnl() -> None:
    portfolio = Portfolio.create(1_000.0)
    portfolio.open_position(
        symbol="BTC/USD",
        side=Side.SHORT,
        qty=1.0,
        price=100.0,
        stop_price=105.0,
        take_profit_price=90.0,
        ts=NOW,
        fee=0.0,
    )
    assert portfolio.equity({"BTC/USD": 95.0}) == pytest.approx(1_005.0)


def test_cannot_open_two_positions_for_one_symbol() -> None:
    portfolio = Portfolio.create(1_000.0)
    kwargs = {
        "symbol": "BTC/USD",
        "side": Side.LONG,
        "qty": 1.0,
        "price": 100.0,
        "stop_price": 95.0,
        "take_profit_price": None,
        "ts": NOW,
        "fee": 0.0,
    }
    portfolio.open_position(**kwargs)  # type: ignore[arg-type]
    with pytest.raises(BrokerError):
        portfolio.open_position(**kwargs)  # type: ignore[arg-type]


def test_closing_nothing_raises() -> None:
    with pytest.raises(BrokerError):
        Portfolio.create(1_000.0).close_position(
            symbol="BTC/USD", price=100.0, ts=NOW, fee=0.0
        )


def test_equity_requires_mark_for_open_position() -> None:
    portfolio = Portfolio.create(1_000.0)
    portfolio.open_position(
        symbol="BTC/USD",
        side=Side.LONG,
        qty=1.0,
        price=100.0,
        stop_price=95.0,
        take_profit_price=None,
        ts=NOW,
        fee=0.0,
    )
    with pytest.raises(KeyError):
        portfolio.equity({})
