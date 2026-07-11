"""Pre-trade risk checks. The same object runs in backtest and live.

A rejected order is logged with a specific reason and dropped. It never
silently vanishes.
"""

from __future__ import annotations

from decimal import Decimal

import structlog

from twenty.backtest.types import PortfolioView, ProposedOrder, Side
from twenty.risk.limits import RiskLimits

log = structlog.get_logger(__name__)


class PreTradeRiskCheck:
    def __init__(self, limits: RiskLimits | None = None) -> None:
        self.limits = limits or RiskLimits()

    def check(
        self, orders: list[ProposedOrder], view: PortfolioView
    ) -> list[ProposedOrder]:
        limits = self.limits
        accepted: list[ProposedOrder] = []
        # Track incremental cash commitment so a batch of buys cannot jointly
        # exceed available cash even if each one individually fits.
        committed = Decimal(0)
        for order in orders:
            if len(accepted) >= limits.max_orders_per_day:
                log.warning(
                    "Order rejected: daily order cap reached",
                    symbol=order.symbol,
                    cap=limits.max_orders_per_day,
                )
                continue
            if order.notional > limits.max_notional_per_order:
                log.warning(
                    "Order rejected: notional above per-order maximum",
                    symbol=order.symbol,
                    notional=str(order.notional),
                    maximum=str(limits.max_notional_per_order),
                )
                continue
            if order.notional < limits.min_notional_per_order:
                log.warning(
                    "Order rejected: notional below per-order minimum",
                    symbol=order.symbol,
                    notional=str(order.notional),
                    minimum=str(limits.min_notional_per_order),
                )
                continue
            if order.side is Side.SELL:
                # Sells precede buys by construction, and execution waits for
                # sell fills before transmitting buys. Credit proceeds at a
                # 2% haircut (commission cap plus spread plus slippage room).
                committed -= order.notional * Decimal("0.98")
            else:
                headroom = view.cash - committed
                # 1% commission cap plus spread: 1.02 covers the worst case.
                required = order.notional * Decimal("1.02")
                if required > headroom:
                    log.warning(
                        "Order rejected: insufficient cash",
                        symbol=order.symbol,
                        required=str(required),
                        headroom=str(headroom),
                    )
                    continue
                committed += required
            accepted.append(order)
        return accepted
