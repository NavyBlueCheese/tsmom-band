"""THE ANTI-LOOKAHEAD SUITE. These five tests define the engine."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from decimal import Decimal

import numpy as np
import polars as pl
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.backtest.conftest import N_DAYS, START, SmaCrossStrategy, make_frame
from twenty.backtest.engine import BacktestEngine
from twenty.backtest.fills import NextBarOpen
from twenty.backtest.ledger import Ledger
from twenty.backtest.types import PortfolioView, ProposedOrder
from twenty.costs.ibkr import order_cost
from twenty.costs.slippage import HalfSpreadCost
from twenty.data.store import Catalog, Snapshot

CAPITAL = Decimal("20.00")


def run(frame: pl.DataFrame, end: datetime | None = None) -> Ledger:
    engine = BacktestEngine(
        catalog=Catalog(frame),
        strategy=SmaCrossStrategy(),
        fill_model=NextBarOpen(),
        initial_capital=CAPITAL,
    )
    return engine.run(end=end)


def test_future_poisoning() -> None:
    """Replacing everything after t with NaN must not change any row <= t."""
    frame = make_frame()
    baseline = run(frame)
    assert len(baseline) == N_DAYS
    for fraction in (0.25, 0.50, 0.75):
        cut_idx = int(N_DAYS * fraction)
        cut_ts = START + timedelta(days=cut_idx)
        poisoned = frame.with_columns(
            [
                pl.when(pl.col("ts") > cut_ts)
                .then(float("nan"))
                .otherwise(pl.col(c))
                .alias(c)
                for c in ("open", "high", "low", "close")
            ]
        )
        rerun = run(poisoned, end=cut_ts)
        for i in range(len(rerun)):
            assert rerun.canonical_row(i) == baseline.canonical_row(i), (
                f"Ledger row {i} changed after poisoning bars beyond {cut_ts}"
            )


def test_shuffled_future() -> None:
    """Permuting all bars after the midpoint must leave the first half
    byte-identical. Ten seeds."""
    frame = make_frame()
    baseline = run(frame)
    mid_idx = N_DAYS // 2
    mid_ts = START + timedelta(days=mid_idx)
    for seed in range(10):
        rng = np.random.default_rng(seed)
        shuffled_parts = []
        for (_,), sym_df in frame.group_by("symbol", maintain_order=True):
            head = sym_df.filter(pl.col("ts") <= mid_ts)
            tail = sym_df.filter(pl.col("ts") > mid_ts)
            perm = rng.permutation(len(tail))
            tail_shuffled = tail.select(
                [pl.col("ts")]
                + [pl.col(c).gather(perm) for c in tail.columns if c != "ts"]
            ).select(tail.columns)
            shuffled_parts.extend([head, tail_shuffled])
        shuffled = pl.concat(shuffled_parts)
        rerun = run(shuffled)
        for i in range(mid_idx + 1):
            assert rerun.canonical_row(i) == baseline.canonical_row(i), (
                f"First-half ledger row {i} changed under future shuffle seed {seed}"
            )


def test_no_same_bar_fill() -> None:
    """Under NextBarOpen no fill may share a timestamp with the snapshot that
    produced its order."""
    ledger = run(make_frame())
    order_ts: dict[str, str] = {}
    for row in ledger.rows:
        for order in json.loads(row["orders"]):
            order_ts[order["client_id"]] = row["ts"]
    fills = ledger.fills()
    assert fills, "Fixture strategy produced no fills; test is vacuous"
    for fill in fills:
        decided = order_ts[fill["client_id"]]
        assert fill["ts"] != decided, f"Fill at {fill['ts']} on its own signal bar"
        assert fill["ts"] > decided


class NullStrategy:
    def __init__(self) -> None:
        self.state: dict[str, object] = {}

    def on_bar(self, snapshot: Snapshot, portfolio: PortfolioView) -> list[ProposedOrder]:
        return []


def test_null_strategy() -> None:
    engine = BacktestEngine(
        catalog=Catalog(make_frame()),
        strategy=NullStrategy(),
        fill_model=NextBarOpen(),
        initial_capital=CAPITAL,
    )
    ledger = engine.run()
    assert len(ledger.fills()) == 0
    assert ledger.total_commission() == Decimal(0)
    assert Decimal(ledger.rows[-1]["cash"]) == CAPITAL
    assert Decimal(ledger.rows[-1]["equity"]) == CAPITAL


@given(notional_cents=st.integers(min_value=200, max_value=50_000))
@settings(max_examples=200, deadline=None)
def test_cost_monotonicity(notional_cents: int) -> None:
    """Doubling an order's notional never decreases total cost."""
    price = Decimal("100.00")
    spread = HalfSpreadCost()

    def total_cost(notional: Decimal) -> Decimal:
        shares = notional / price
        return order_cost(shares, price, is_sell=False) + spread.cost("SPY", notional)

    notional = Decimal(notional_cents) / 100
    assert total_cost(notional * 2) >= total_cost(notional)
