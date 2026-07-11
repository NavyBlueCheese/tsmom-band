"""Strategy tests: index arithmetic, band behaviour, calendar, ordering."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal

import exchange_calendars as xcals
import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from twenty.backtest.types import PortfolioView, Side
from twenty.data.store import Catalog, Snapshot
from twenty.strategies.tsmom_band import (
    LOOKBACK,
    SKIP,
    UNIVERSE,
    TsmomBand,
    is_quarter_end_session,
    momentum,
    signal,
    target_weights,
)

N_SESSIONS = 320  # > LOOKBACK + SKIP + 1


def _sessions(n: int, end: str = "2018-06-29") -> list[datetime]:
    """The last ``n`` real XNYS sessions ending at ``end`` (a quarter-end),
    as UTC-midnight timestamps."""
    cal = xcals.get_calendar("XNYS")
    sessions = cal.sessions_in_range("2014-01-01", end)
    return [
        datetime(s.year, s.month, s.day, tzinfo=UTC) for s in sessions[-n:]
    ]


def make_snapshot(
    prices: dict[str, Sequence[float]], n: int = N_SESSIONS, end: str = "2018-06-29"
) -> Snapshot:
    ts = _sessions(n, end)
    frames = []
    for symbol in UNIVERSE:
        p = np.asarray(prices[symbol], dtype=np.float64)
        assert p.shape[0] == n
        frames.append(
            pl.DataFrame(
                {
                    "symbol": [symbol] * n,
                    "ts": ts,
                    "open": p,
                    "high": p,
                    "low": p,
                    "close": p,
                    "volume": [1e6] * n,
                    "adj_factor": [1.0] * n,
                    "dividend": [0.0] * n,
                    "split": [1.0] * n,
                }
            )
        )
    catalog = Catalog(pl.concat(frames))
    return catalog.as_of(ts[-1])


def _wiggly(n: int, base: float, seed: int, drift: float = 0.0002) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return base * np.exp(np.cumsum(rng.normal(drift, 0.01, n)))


def default_prices(n: int = N_SESSIONS) -> dict[str, np.ndarray]:
    return {s: _wiggly(n, 50.0 + 10 * i, seed=i) for i, s in enumerate(UNIVERSE)}


def flat_view() -> PortfolioView:
    return PortfolioView(positions={}, cash=Decimal("20.00"))


# --- momentum index arithmetic -------------------------------------------------


def test_momentum_hand_computed_fixture() -> None:
    """Prices are the session index itself, so momentum is fully hand-computable.

    With n sessions, P[-1] = n-1 + 100. The momentum numerator is bar t-21
    (value n-22+100) and the denominator bar t-252 (value n-253+100).
    """
    n = N_SESSIONS
    ramp = np.arange(n, dtype=np.float64) + 100.0
    prices = {s: ramp for s in UNIVERSE}
    mom = momentum(make_snapshot(prices))
    numerator = (n - 1) - SKIP + 100.0  # value at bar t-21
    denominator = (n - 1) - 252 + 100.0  # value at bar t-252
    expected = numerator / denominator - 1.0
    for symbol in UNIVERSE:
        assert mom[symbol] == pytest.approx(expected, abs=1e-12)


def test_momentum_skips_final_month() -> None:
    """A price move inside the last SKIP bars must not change momentum."""
    n = N_SESSIONS
    base = np.full(n, 100.0)
    base[: n - 300] = 80.0
    bumped = base.copy()
    bumped[-SKIP:] *= 1.5  # move entirely inside the skip window
    mom_base = momentum(make_snapshot({s: base for s in UNIVERSE}))
    mom_bump = momentum(make_snapshot({s: bumped for s in UNIVERSE}))
    assert mom_base == mom_bump


def test_signal_monotone_up_spy_only() -> None:
    n = N_SESSIONS
    prices: dict[str, np.ndarray] = {s: np.full(n, 100.0) for s in UNIVERSE}
    prices["SPY"] = np.linspace(100.0, 200.0, n)
    sig = signal(make_snapshot(prices))
    assert sig == {"SPY": 1, "EFA": 0, "IEF": 0, "GLD": 0}


# --- the band ------------------------------------------------------------------


def _snapshot_and_target() -> tuple[Snapshot, dict[str, Decimal]]:
    snap = make_snapshot(default_prices())
    return snap, target_weights(snap)


def _view_with_weights(
    snap: Snapshot, weights: dict[str, Decimal], capital: Decimal = Decimal("20.00")
) -> PortfolioView:
    positions: dict[str, Decimal] = {}
    cash = capital
    for symbol, w in weights.items():
        mark = Decimal(str(float(snap.raw_close(symbol)[-1])))
        shares = (w * capital) / mark
        positions[symbol] = shares
        cash -= shares * mark
    return PortfolioView(positions=positions, cash=cash)


def test_no_orders_when_current_equals_target() -> None:
    snap, target = _snapshot_and_target()
    view = _view_with_weights(snap, target)
    assert TsmomBand().on_bar(snap, view) == []


def test_nineteen_point_deviation_no_trade() -> None:
    snap, target = _snapshot_and_target()
    held = dict(target)
    long_legs = [s for s in UNIVERSE if target[s] > Decimal("0.2")]
    assert long_legs, "Fixture produced no long leg large enough to perturb"
    leg = long_legs[0]
    held[leg] = target[leg] - Decimal("0.19")
    view = _view_with_weights(snap, held)
    assert TsmomBand().on_bar(snap, view) == []


def test_twenty_one_point_deviation_trades_that_leg_only() -> None:
    snap, target = _snapshot_and_target()
    held = dict(target)
    long_legs = [s for s in UNIVERSE if target[s] > Decimal("0.25")]
    assert long_legs, "Fixture produced no long leg large enough to perturb"
    leg = long_legs[0]
    held[leg] = target[leg] - Decimal("0.21")
    view = _view_with_weights(snap, held)
    orders = TsmomBand().on_bar(snap, view)
    assert len(orders) == 1
    assert orders[0].symbol == leg
    assert orders[0].side is Side.BUY


# --- calendar ------------------------------------------------------------------


def test_non_quarter_end_returns_empty_regardless_of_weights() -> None:
    snap = make_snapshot(default_prices(), end="2018-06-20")  # mid-quarter session
    view = flat_view()
    assert TsmomBand().on_bar(snap, view) == []


def test_2026_quarter_end_sessions_match_calendar() -> None:
    """The 2026 quarter ends, from the XNYS calendar itself. In particular
    Q4 2026: December 31 2026 is a Thursday; whether it is the last session
    is the calendar's call, not date arithmetic's."""
    cal = xcals.get_calendar("XNYS")
    for month_end in ("2026-03-31", "2026-06-30", "2026-09-30", "2026-12-31"):
        expected = cal.date_to_session(month_end, direction="previous")
        expected_dt = datetime(
            expected.year, expected.month, expected.day, tzinfo=UTC
        )
        assert is_quarter_end_session(expected_dt)
        # The session after the quarter's last session is never a quarter end.
        following = cal.next_session(expected)
        following_dt = datetime(
            following.year, following.month, following.day, tzinfo=UTC
        )
        assert not is_quarter_end_session(following_dt)
    # A midweek session in the middle of a quarter is never a quarter end.
    assert not is_quarter_end_session(datetime(2026, 2, 11, tzinfo=UTC))
    # July 3 2026 is the Independence Day observance; June 30 remains Q2 end.
    assert not is_quarter_end_session(datetime(2026, 7, 2, tzinfo=UTC))


# --- sizing --------------------------------------------------------------------


@given(seed=st.integers(min_value=0, max_value=10_000))
@settings(max_examples=30, deadline=None)
def test_leverage_never_exceeds_one(seed: int) -> None:
    """Over random return-generating PSD covariances (A.T @ A), the vol-target
    scalar k never pushes gross weights above 1."""
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 0.02, (4, 4))
    cov = a.T @ a + np.eye(4) * 1e-6
    chol = np.linalg.cholesky(cov)
    n = N_SESSIONS
    rets = rng.normal(0.0005, 1.0, (n - 1, 4)) @ chol.T
    prices = {
        s: 100.0 * np.exp(np.concatenate([[0.0], np.cumsum(rets[:, i])]))
        for i, s in enumerate(UNIVERSE)
    }
    weights = target_weights(make_snapshot(prices))
    gross = sum(weights.values())
    assert gross <= Decimal("1.0") + Decimal("1e-9")
    assert all(w >= 0 for w in weights.values())


def test_sells_precede_buys() -> None:
    snap, target = _snapshot_and_target()
    held = {s: Decimal(0) for s in UNIVERSE}
    long_legs = [s for s in UNIVERSE if target[s] > Decimal("0.25")]
    assert long_legs
    # Hold a large position in something the target does not want, and
    # nothing in a leg the target wants: forces one sell and one buy.
    short_leg = next(s for s in UNIVERSE if target[s] < Decimal("0.05"))
    held[short_leg] = Decimal("0.60")
    view = _view_with_weights(snap, held)
    orders = TsmomBand().on_bar(snap, view)
    assert len(orders) >= 2
    sides = [o.side for o in orders]
    first_buy = sides.index(Side.BUY) if Side.BUY in sides else len(sides)
    assert all(s is Side.SELL for s in sides[:first_buy])
    assert all(s is Side.BUY for s in sides[first_buy:])


def test_insufficient_history_returns_empty_and_logs() -> None:
    n = LOOKBACK + SKIP  # one short of the minimum
    prices = {s: np.linspace(100, 150, n) for s in UNIVERSE}
    snap = make_snapshot(prices, n=n)
    result = TsmomBand().on_bar(snap, flat_view())
    assert result == []


def test_band_zero_rebalances_everything() -> None:
    snap, target = _snapshot_and_target()
    view = flat_view()
    orders = TsmomBand(band=0.0).on_bar(snap, view)
    wanted = [s for s in UNIVERSE if target[s] * Decimal("20") >= Decimal("2.00")]
    assert sorted(o.symbol for o in orders) == sorted(wanted)
