"""Records crossing the strategy/engine boundary."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Protocol
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from twenty.data.store import Snapshot


class Side(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(StrEnum):
    MARKETABLE_LIMIT = "MARKETABLE_LIMIT"
    MOC = "MOC"


class ProposedOrder(BaseModel):
    model_config = ConfigDict(frozen=True)

    symbol: str
    side: Side
    notional: Decimal
    order_type: OrderType = OrderType.MARKETABLE_LIMIT
    limit_offset_bps: int = 3
    client_id: UUID = Field(default_factory=uuid4)


class Fill(BaseModel):
    model_config = ConfigDict(frozen=True)

    symbol: str
    side: Side
    shares: Decimal
    price: Decimal
    ts: datetime
    commission: Decimal
    slippage: Decimal
    client_id: UUID


class PortfolioView(BaseModel):
    """Read-only picture of the portfolio handed to strategies and risk.

    ``positions`` maps symbol to share count. All values are copies; mutating
    this object does not touch the engine's books, and the model is frozen
    anyway.
    """

    model_config = ConfigDict(frozen=True)

    positions: dict[str, Decimal]
    cash: Decimal

    def total_value(self, marks: dict[str, Decimal]) -> Decimal:
        value = self.cash
        for symbol, shares in self.positions.items():
            if shares != 0:
                value += shares * marks[symbol]
        return value

    def weights(self, marks: dict[str, Decimal]) -> dict[str, Decimal]:
        total = self.total_value(marks)
        if total <= 0:
            return {s: Decimal(0) for s in self.positions}
        return {
            symbol: (shares * marks.get(symbol, Decimal(0)) / total)
            for symbol, shares in self.positions.items()
        }


class Strategy(Protocol):
    """A strategy sees one snapshot and one portfolio view per bar, nothing
    else. Any state it keeps lives in an explicit ``self.state`` dict that it
    owns and the engine never touches."""

    state: dict[str, object]

    def on_bar(self, snapshot: Snapshot, portfolio: PortfolioView) -> list[ProposedOrder]: ...


class RiskCheck(Protocol):
    def check(self, orders: list[ProposedOrder], view: PortfolioView) -> list[ProposedOrder]: ...


class NullRiskCheck:
    """Pass-through used until the risk stage is wired in."""

    def check(self, orders: list[ProposedOrder], view: PortfolioView) -> list[ProposedOrder]:
        return orders
