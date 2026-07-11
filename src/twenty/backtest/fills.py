"""Fill models.

NextBarOpen is the default and the only model used for reported results.
"""

from __future__ import annotations

from datetime import datetime
from decimal import ROUND_DOWN, Decimal
from typing import Protocol

import structlog

from twenty.backtest.types import Fill, ProposedOrder, Side
from twenty.costs.ibkr import order_cost
from twenty.costs.slippage import HalfSpreadCost, assert_impact_immaterial

log = structlog.get_logger(__name__)

_SHARE_QUANTUM = Decimal("0.0001") 


class FillModel(Protocol):
    def fill(
        self,
        orders: list[ProposedOrder],
        prices: dict[str, Decimal],
        ts: datetime,
        available_cash: Decimal,
        held: dict[str, Decimal],
    ) -> list[Fill]: ...


def _execute(
    orders: list[ProposedOrder],
    prices: dict[str, Decimal],
    ts: datetime,
    available_cash: Decimal,
    held: dict[str, Decimal],
    spread: HalfSpreadCost,
) -> list[Fill]:
    """Shared fill mechanics. The reference price is taken as given for the
    session; a marketable limit at mid +/- offset is assumed to fill at that
    reference plus half the spread. Deliberately, nothing here inspects any
    price to decide *whether* the order would have been marketable, using
    the fill bar's price for that decision would be a limit-order lookahead.
    """
    fills: list[Fill] = []
    cash = available_cash
    for order in orders:
        price = prices.get(order.symbol)
        if price is None or price <= 0:
            log.warning("No price at fill session, order dropped", symbol=order.symbol)
            continue
        assert_impact_immaterial(order.symbol, float(order.notional))
        if order.side is Side.SELL:
            shares = (order.notional / price).quantize(_SHARE_QUANTUM, rounding=ROUND_DOWN)
            shares = min(shares, held.get(order.symbol, Decimal(0)))
            if shares <= 0:
                continue
            notional = shares * price
            cost = order_cost(shares, price, is_sell=True)
            slip = spread.cost(order.symbol, notional)
            cash += notional - cost - slip
            held[order.symbol] = held.get(order.symbol, Decimal(0)) - shares
        else:
            shares = (order.notional / price).quantize(_SHARE_QUANTUM, rounding=ROUND_DOWN)
            if shares <= 0:
                continue
            notional = shares * price
            cost = order_cost(shares, price, is_sell=False)
            slip = spread.cost(order.symbol, notional)
            total_needed = notional + cost + slip
            if total_needed > cash:
                # Cash account
                affordable = (cash / (price * Decimal("1.0102"))).quantize(
                    _SHARE_QUANTUM, rounding=ROUND_DOWN
                )
                if affordable <= 0:
                    log.warning("Insufficient cash, order dropped", symbol=order.symbol)
                    continue
                shares = affordable
                notional = shares * price
                cost = order_cost(shares, price, is_sell=False)
                slip = spread.cost(order.symbol, notional)
            cash -= notional + cost + slip
            held[order.symbol] = held.get(order.symbol, Decimal(0)) + shares
        fills.append(
            Fill(
                symbol=order.symbol,
                side=order.side,
                shares=shares,
                price=price,
                ts=ts,
                commission=cost,
                slippage=slip,
                client_id=order.client_id,
            )
        )
    return fills


class NextBarOpen:
    """Signal on bar t's close, filled at bar t+1's open. Default, and the
    only fill model used for reported results."""

    def __init__(self, spread: HalfSpreadCost | None = None) -> None:
        self._spread = spread or HalfSpreadCost()

    def fill(
        self,
        orders: list[ProposedOrder],
        prices: dict[str, Decimal],
        ts: datetime,
        available_cash: Decimal,
        held: dict[str, Decimal],
    ) -> list[Fill]:
        return _execute(orders, prices, ts, available_cash, held, self._spread)


class SameBarClose:
    """Filled at bar t's own close. OPTIMISTIC: it assumes an MOC order was
    live before the signal bar completed, which overstates achievable fills.
    Never used for reported results; selecting it requires the explicit
    ``i_understand_this_is_optimistic`` flag."""

    def __init__(
        self,
        spread: HalfSpreadCost | None = None,
        i_understand_this_is_optimistic: bool = False,
    ) -> None:
        if not i_understand_this_is_optimistic:
            raise ValueError(
                "SameBarClose is optimistic; pass i_understand_this_is_optimistic=True"
            )
        self._spread = spread or HalfSpreadCost()

    def fill(
        self,
        orders: list[ProposedOrder],
        prices: dict[str, Decimal],
        ts: datetime,
        available_cash: Decimal,
        held: dict[str, Decimal],
    ) -> list[Fill]:
        return _execute(orders, prices, ts, available_cash, held, self._spread)
