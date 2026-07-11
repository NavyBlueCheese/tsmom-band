"""IBBroker: the live implementation of the Broker protocol.

Orders are cash-quantity (dollar-denominated fractional) limit orders at
mid +/- 3bps. Every raw IBKR error code is logged, always, including the ones
classified as warnings — the warning set is exactly where ib_insync went
stale, and silent codes are how money disappears.
"""

from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal
from typing import Any, Protocol
from uuid import UUID

import structlog

from twenty.backtest.types import ProposedOrder, Side
from twenty.execution.journal import Journal, OrderState

log = structlog.get_logger(__name__)

# IBKR error codes that are informational in normal operation. They are still
# logged at full fidelity; classification only affects alerting, never logging.
INFORMATIONAL_CODES = frozenset({2104, 2106, 2158, 2119})
# Codes where the client sees an error but the order may remain live at the
# broker. Never retry these blindly; reconcile first. 10349 is the canonical
# example.
ORDER_MAY_BE_LIVE_CODES = frozenset({10349, 10148, 202})


def _usable_mid(ticker: Any) -> float | None:
    """Mid from bid/ask when both sides are live, else last, else close.
    NaN-safe: delayed tickers report nan until data arrives."""

    def good(x: Any) -> bool:
        return x is not None and isinstance(x, (int, float)) and x > 0 and x == x

    if good(ticker.bid) and good(ticker.ask):
        return (float(ticker.bid) + float(ticker.ask)) / 2.0
    if good(ticker.last):
        return float(ticker.last)
    if good(ticker.close):
        return float(ticker.close)
    return None


class Broker(Protocol):
    """The protocol both the backtest fill model adapter and IBBroker satisfy."""

    def place_order(self, order: ProposedOrder) -> None: ...
    def cancel_order(self, client_id: UUID) -> None: ...
    def get_positions(self) -> dict[str, Decimal]: ...
    def get_account_value(self) -> Decimal: ...
    def subscribe_fills(self, callback: Callable[[dict[str, Any]], None]) -> None: ...


class IBBroker:
    def __init__(self, ib: Any, journal: Journal, account: str) -> None:
        self._ib = ib
        self._journal = journal
        self._account = account
        self._fill_callbacks: list[Callable[[dict[str, Any]], None]] = []
        self._trades_by_client_id: dict[UUID, Any] = {}
        self.consecutive_rejections = 0
        ib.errorEvent += self._on_error
        ib.orderStatusEvent += self._on_order_status
        ib.execDetailsEvent += self._on_exec_details

    # -- events ------------------------------------------------------------

    def _on_error(
        self, req_id: int, code: int, message: str, contract: Any = None
    ) -> None:
        # Log every raw code and message, always, even the "warnings".
        log.warning(
            "IBKR message",
            req_id=req_id,
            code=code,
            message=message,
            informational=code in INFORMATIONAL_CODES,
            order_may_be_live=code in ORDER_MAY_BE_LIVE_CODES,
        )
        if code in ORDER_MAY_BE_LIVE_CODES:
            # The order may still be working at the broker. The journal keeps
            # it TRANSMITTED (non-terminal), so may_transmit refuses any
            # retry until a reconcile settles its true state.
            log.warning(
                "Order state ambiguous after broker error; retry is forbidden "
                "until reconciliation",
                code=code,
            )

    def _on_order_status(self, trade: Any) -> None:
        status = trade.orderStatus.status
        client_id = self._client_id_for(trade)
        if client_id is None:
            return
        if status in ("Filled",):
            self._journal.set_state(client_id, OrderState.FILLED)
            self.consecutive_rejections = 0
        elif status in ("Cancelled", "ApiCancelled"):
            self._journal.set_state(client_id, OrderState.CANCELLED)
        elif status in ("Inactive",):
            self._journal.set_state(client_id, OrderState.REJECTED)
            self.consecutive_rejections += 1
            log.warning(
                "Order rejected by broker",
                client_id=str(client_id),
                streak=self.consecutive_rejections,
            )

    def _on_exec_details(self, trade: Any, fill: Any) -> None:
        record = {
            "symbol": fill.contract.symbol,
            "shares": str(fill.execution.shares),
            "price": str(fill.execution.price),
            "time": str(fill.execution.time),
        }
        log.info("Fill", **record)
        for callback in self._fill_callbacks:
            callback(record)

    def _client_id_for(self, trade: Any) -> UUID | None:
        for client_id, known in self._trades_by_client_id.items():
            if known.order.orderId == trade.order.orderId:
                return client_id
        return None

    # -- Broker protocol -----------------------------------------------------

    def place_order(self, order: ProposedOrder) -> None:
        from ib_async import LimitOrder, Stock

        if not self._journal.may_transmit(order):
            return
        self._journal.record_proposed(order)

        contract = Stock(order.symbol, "SMART", "USD")
        qualified = self._ib.qualifyContracts(contract)
        if not qualified:
            log.warning("Contract failed to qualify", symbol=order.symbol)
            self._journal.set_state(order.client_id, OrderState.CANCELLED)
            return
        contract = qualified[0]
        # Delayed data (type 3) is requested up front: paper accounts often
        # carry no live subscription, and IBKR then serves delayed quotes
        # instead of erroring. With a live subscription this is upgraded to
        # real-time automatically.
        self._ib.reqMarketDataType(3)
        ticker = self._ib.reqMktData(contract, "", False, False)
        deadline = 10.0
        while deadline > 0 and not _usable_mid(ticker):
            self._ib.sleep(0.5)
            deadline -= 0.5
        mid_f = _usable_mid(ticker)
        if mid_f is None:
            log.warning("No usable quote, order not transmitted", symbol=order.symbol)
            self._journal.set_state(order.client_id, OrderState.CANCELLED)
            return
        mid = Decimal(str(mid_f))
        offset = mid * Decimal(order.limit_offset_bps) / Decimal(10_000)
        if order.side is Side.BUY:
            limit_price = (mid + offset).quantize(Decimal("0.01"))
        else:
            limit_price = (mid - offset).quantize(Decimal("0.01"))

        ib_order = LimitOrder(
            action=order.side.value,
            totalQuantity=0,
            lmtPrice=float(limit_price),
        )
        ib_order.cashQty = float(order.notional)  # dollar-denominated fractional
        ib_order.tif = "DAY"
        ib_order.orderRef = str(order.client_id)

        trade = self._ib.placeOrder(contract, ib_order)
        self._trades_by_client_id[order.client_id] = trade
        self._journal.set_state(
            order.client_id, OrderState.TRANSMITTED, broker_order_id=trade.order.orderId
        )
        log.info(
            "Order transmitted",
            symbol=order.symbol,
            side=order.side.value,
            notional=str(order.notional),
            limit=str(limit_price),
            client_id=str(order.client_id),
        )

    def cancel_order(self, client_id: UUID) -> None:
        trade = self._trades_by_client_id.get(client_id)
        if trade is not None:
            self._ib.cancelOrder(trade.order)

    def cancel_all_open(self) -> None:
        for trade in self._ib.openTrades():
            self._ib.cancelOrder(trade.order)

    def get_positions(self) -> dict[str, Decimal]:
        out: dict[str, Decimal] = {}
        for position in self._ib.positions(self._account):
            out[position.contract.symbol] = Decimal(str(position.position))
        return out

    def get_account_value(self) -> Decimal:
        for row in self._ib.accountSummary(self._account):
            if row.tag == "NetLiquidation":
                return Decimal(row.value)
        raise RuntimeError("NetLiquidation not present in account summary")

    def get_cash(self) -> Decimal:
        for row in self._ib.accountSummary(self._account):
            if row.tag == "TotalCashValue":
                return Decimal(row.value)
        raise RuntimeError("TotalCashValue not present in account summary")

    def subscribe_fills(self, callback: Callable[[dict[str, Any]], None]) -> None:
        self._fill_callbacks.append(callback)
