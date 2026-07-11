"""The event engine. Correctness > auditability > speed. Never vectorized.

Per-session order of operations (session t):
  1. Corporate actions dated t: dividends credited to cash and splits applied
     to share counts, for positions held coming into t. Fills occurring at
     t's open are applied *after* actions, so an ex-date buyer does not
     receive the dividend.
  2. Pending fills — orders decided at t-1, filled at t's open (NextBarOpen).
  3. snapshot = catalog.as_of(t); strategy.on_bar; risk check; the resulting
     orders become pending for t+1.
  4. Mark at t's close, assert the accounting invariant, write the ledger row.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import structlog

from twenty.backtest.fills import FillModel, SameBarClose
from twenty.backtest.ledger import Ledger
from twenty.backtest.types import (
    Fill,
    NullRiskCheck,
    PortfolioView,
    ProposedOrder,
    RiskCheck,
    Side,
    Strategy,
)
from twenty.data.store import Catalog

log = structlog.get_logger(__name__)

_TOL = Decimal("0.00000001")


class Portfolio:
    """The engine's books. Strategies only ever see PortfolioView copies."""

    def __init__(self, initial_capital: Decimal) -> None:
        self.cash = initial_capital
        self.positions: dict[str, Decimal] = {}

    def view(self) -> PortfolioView:
        return PortfolioView(positions=dict(self.positions), cash=self.cash)

    def apply(self, fills: list[Fill]) -> None:
        for fill in fills:
            gross = fill.shares * fill.price
            if fill.side is Side.BUY:
                self.cash -= gross + fill.commission + fill.slippage
                self.positions[fill.symbol] = (
                    self.positions.get(fill.symbol, Decimal(0)) + fill.shares
                )
            else:
                self.cash += gross - fill.commission - fill.slippage
                self.positions[fill.symbol] = (
                    self.positions.get(fill.symbol, Decimal(0)) - fill.shares
                )


class BacktestEngine:
    def __init__(
        self,
        catalog: Catalog,
        strategy: Strategy,
        fill_model: FillModel,
        initial_capital: Decimal,
        risk: RiskCheck | None = None,
    ) -> None:
        self.catalog = catalog
        self.strategy = strategy
        self.fill_model = fill_model
        self.initial_capital = initial_capital
        self.risk = risk if risk is not None else NullRiskCheck()
        self._same_bar = isinstance(fill_model, SameBarClose)

    def _marks(self, portfolio: Portfolio, ts: datetime) -> dict[str, Decimal]:
        marks: dict[str, Decimal] = {}
        for symbol, shares in portfolio.positions.items():
            if shares == 0:
                continue
            close = self.catalog.close_at(symbol, ts)
            if close is None:
                raise RuntimeError(f"No mark for held position {symbol} at {ts}")
            marks[symbol] = Decimal(str(close))
        return marks

    def run(self, start: datetime | None = None, end: datetime | None = None) -> Ledger:
        portfolio = Portfolio(self.initial_capital)
        ledger = Ledger()
        pending: list[ProposedOrder] = []
        expected_cash = self.initial_capital
        prev_equity = self.initial_capital
        cumulative_pnl = Decimal(0)

        for ts in self.catalog.sessions():
            if start is not None and ts < start:
                continue
            if end is not None and ts > end:
                break

            # Corporate actions on positions held coming into t.
            for symbol in list(portfolio.positions):
                shares = portfolio.positions[symbol]
                if shares == 0:
                    continue
                dividend, split = self.catalog.actions_at(symbol, ts)
                if dividend != 0.0:
                    credit = shares * Decimal(str(dividend))
                    portfolio.cash += credit
                    expected_cash += credit
                if split != 1.0:
                    portfolio.positions[symbol] = shares * Decimal(str(split))

            # Fills for orders decided at the previous session (t-1 close),
            #    executed at t's open.
            session_fills: list[Fill] = []
            if pending and not self._same_bar:
                prices: dict[str, Decimal] = {}
                for order in pending:
                    px = self.catalog.open_at(order.symbol, ts)
                    if px is not None:
                        prices[order.symbol] = Decimal(str(px))
                session_fills = self.fill_model.fill(
                    pending, prices, ts, portfolio.cash, dict(portfolio.positions)
                )
                portfolio.apply(session_fills)
                pending = []

            # Decide. The strategy sees only the snapshot and a view.
            snapshot = self.catalog.as_of(ts)
            view = portfolio.view()
            orders = self.strategy.on_bar(snapshot, view)
            orders = self.risk.check(orders, view)

            if self._same_bar and orders:
                prices = {}
                for order in orders:
                    px = self.catalog.close_at(order.symbol, ts)
                    if px is not None:
                        prices[order.symbol] = Decimal(str(px))
                same_bar_fills = self.fill_model.fill(
                    orders, prices, ts, portfolio.cash, dict(portfolio.positions)
                )
                portfolio.apply(same_bar_fills)
                session_fills = session_fills + same_bar_fills
            elif orders:
                pending = list(orders)

            # Mark, assert, record.
            for fill in session_fills:
                flow = fill.shares * fill.price + fill.commission + fill.slippage
                if fill.side is Side.BUY:
                    expected_cash -= flow
                else:
                    expected_cash += (
                        fill.shares * fill.price - fill.commission - fill.slippage
                    )
            if abs(portfolio.cash - expected_cash) > _TOL:
                raise AssertionError(
                    f"Cash reconstruction mismatch at {ts}: book {portfolio.cash} "
                    f"vs reconstructed {expected_cash}"
                )
            marks = self._marks(portfolio, ts)
            market_value = sum(
                (shares * marks[symbol] for symbol, shares in portfolio.positions.items()
                if shares != 0),
                Decimal(0),
            )
            equity = portfolio.cash + market_value
            cumulative_pnl += equity - prev_equity
            prev_equity = equity
            if abs((expected_cash + market_value)
                - (self.initial_capital + cumulative_pnl)) > _TOL:
                raise AssertionError(
                    f"Accounting invariant violated at {ts}: cash+mv "
                    f"{expected_cash + market_value} vs initial+pnl "
                    f"{self.initial_capital + cumulative_pnl}"
                )
            ledger.record(
                ts=ts,
                snapshot_hash=snapshot.hash(),
                orders=list(orders),
                fills=session_fills,
                cash=portfolio.cash,
                positions=dict(portfolio.positions),
                equity=equity,
            )
        return ledger
