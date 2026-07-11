from __future__ import annotations

from decimal import Decimal

import pytest

from twenty.backtest.types import PortfolioView, ProposedOrder, Side
from twenty.risk.checks import PreTradeRiskCheck
from twenty.risk.limits import RiskLimits
from twenty.risk.sizing import assert_no_leverage


def view(cash: str = "20.00") -> PortfolioView:
    return PortfolioView(positions={}, cash=Decimal(cash))


def order(symbol: str, side: Side, notional: str) -> ProposedOrder:
    return ProposedOrder(symbol=symbol, side=side, notional=Decimal(notional))


def test_min_notional_rejected() -> None:
    check = PreTradeRiskCheck()
    out = check.check([order("SPY", Side.BUY, "1.80")], view())
    assert out == []


def test_max_notional_rejected() -> None:
    check = PreTradeRiskCheck()
    out = check.check([order("SPY", Side.BUY, "16.00")], view())
    assert out == []


def test_order_cap() -> None:
    check = PreTradeRiskCheck(RiskLimits(max_orders_per_day=2))
    orders = [order(f"S{i}", Side.BUY, "3.00") for i in range(4)]
    out = check.check(orders, view())
    assert len(out) == 2


def test_insufficient_cash_rejected() -> None:
    check = PreTradeRiskCheck()
    out = check.check([order("SPY", Side.BUY, "15.00")], view("5.00"))
    assert out == []


def test_buy_funded_by_preceding_sell_accepted() -> None:
    check = PreTradeRiskCheck()
    orders = [order("IEF", Side.SELL, "10.00"), order("SPY", Side.BUY, "12.00")]
    out = check.check(orders, view("3.00"))
    assert [o.symbol for o in out] == ["IEF", "SPY"]


def test_accepted_orders_pass_through_unchanged() -> None:
    check = PreTradeRiskCheck()
    orders = [order("SPY", Side.BUY, "5.00")]
    assert check.check(orders, view()) == orders


def test_assert_no_leverage() -> None:
    assert_no_leverage({"SPY": Decimal("0.5"), "IEF": Decimal("0.5")})
    with pytest.raises(AssertionError):
        assert_no_leverage({"SPY": Decimal("0.7"), "IEF": Decimal("0.4")})
