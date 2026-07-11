"""Shared fixtures: a deterministic two-symbol market and a history-dependent
strategy that trades often enough to make lookahead detectable."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import numpy as np
import polars as pl
import pytest

from twenty.backtest.types import PortfolioView, ProposedOrder, Side
from twenty.data.store import Catalog, Snapshot

START = datetime(2015, 1, 5, tzinfo=UTC)
N_DAYS = 240
SYMBOLS = ("AAA", "BBB")


def make_frame(seed: int = 7) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for j, symbol in enumerate(SYMBOLS):
        ts = [START + timedelta(days=i) for i in range(N_DAYS)]
        rets = rng.normal(0.0004, 0.012, N_DAYS)
        close = 50.0 * (1 + j) * np.exp(np.cumsum(rets))
        open_ = close * (1 + rng.normal(0, 0.002, N_DAYS))
        frames.append(
            pl.DataFrame(
                {
                    "symbol": [symbol] * N_DAYS,
                    "ts": ts,
                    "open": open_,
                    "high": np.maximum(open_, close) * 1.005,
                    "low": np.minimum(open_, close) * 0.995,
                    "close": close,
                    "volume": [1e6] * N_DAYS,
                    "adj_factor": [1.0] * N_DAYS,
                    "dividend": [0.0] * N_DAYS,
                    "split": [1.0] * N_DAYS,
                }
            )
        )
    return pl.concat(frames)


class SmaCrossStrategy:
    """Long $6 of a symbol when close > 10-bar SMA, flat otherwise. History
    dependent, so poisoned or shuffled futures change its decisions."""

    def __init__(self) -> None:
        self.state: dict[str, object] = {}

    def on_bar(self, snapshot: Snapshot, portfolio: PortfolioView) -> list[ProposedOrder]:
        orders: list[ProposedOrder] = []
        for symbol in snapshot.symbols():
            closes = snapshot.adjusted_close(symbol)
            if closes.shape[0] < 10:
                continue
            sma = float(closes[-10:].mean())
            last = float(closes[-1])
            held = portfolio.positions.get(symbol, Decimal(0))
            if last > sma and held == 0:
                orders.append(
                    ProposedOrder(symbol=symbol, side=Side.BUY, notional=Decimal("6.00"))
                )
            elif last <= sma and held > 0:
                mark = Decimal(str(last))
                orders.append(
                    ProposedOrder(symbol=symbol, side=Side.SELL, notional=held * mark)
                )
        return orders


@pytest.fixture()
def catalog() -> Catalog:
    return Catalog(make_frame())


@pytest.fixture()
def strategy() -> SmaCrossStrategy:
    return SmaCrossStrategy()
