"""Property tests for the point-in-time store."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from twenty.data.store import Catalog

START = datetime(2020, 1, 1, tzinfo=UTC)
N_DAYS = 120
SPLIT_DAY = 60  # index of the 2:1 split
DIV_DAY = 30
DIV_AMOUNT = 0.50


def _fixture_frame() -> pl.DataFrame:
    """One symbol with a known $0.50 dividend at day 30 and a 2:1 split at
    day 60, raw prices halving at the split as they would in reality."""
    ts = [START + timedelta(days=i) for i in range(N_DAYS)]
    base = 100.0 + np.cumsum(np.sin(np.arange(N_DAYS)) * 0.5)
    close = base.copy()
    close[SPLIT_DAY:] = close[SPLIT_DAY:] / 2.0  # raw price halves at the split
    factor = np.ones(N_DAYS)
    factor[DIV_DAY:] *= (close[DIV_DAY] + DIV_AMOUNT) / close[DIV_DAY]
    factor[SPLIT_DAY:] *= 2.0
    dividend = np.zeros(N_DAYS)
    dividend[DIV_DAY] = DIV_AMOUNT
    split = np.ones(N_DAYS)
    split[SPLIT_DAY] = 2.0
    return pl.DataFrame(
        {
            "symbol": ["TST"] * N_DAYS,
            "ts": ts,
            "open": close * 0.999,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": [1_000.0] * N_DAYS,
            "adj_factor": factor,
            "dividend": dividend,
            "split": split,
        }
    )


@pytest.fixture(scope="module")
def catalog() -> Catalog:
    return Catalog(_fixture_frame())


@given(offset_days=st.integers(min_value=0, max_value=N_DAYS + 30))
@settings(max_examples=60, deadline=None)
def test_as_of_never_contains_future_rows(offset_days: int) -> None:
    catalog = Catalog(_fixture_frame())
    ts = START + timedelta(days=offset_days)
    snap = catalog.as_of(ts)
    cutoff_ns = int(ts.timestamp() * 1e9)
    for symbol in snap.symbols():
        ts_ns = snap._data[symbol].ts_ns
        assert (ts_ns <= cutoff_ns).all()


def test_adjusted_returns_match_vendor_adjusted_series(catalog: Catalog) -> None:
    """Raw close + adj_factor -> adjusted close -> returns must equal returns
    from a vendor-style fully adjusted series, across the 2:1 split."""
    df = _fixture_frame()
    close = df["close"].to_numpy()
    factor = df["adj_factor"].to_numpy()
    vendor_adjusted = close * factor / factor[-1]
    vendor_returns = vendor_adjusted[1:] / vendor_adjusted[:-1] - 1.0

    snap = catalog.as_of(START + timedelta(days=N_DAYS))
    ours = snap.adjusted_close("TST")
    our_returns = ours[1:] / ours[:-1] - 1.0
    np.testing.assert_allclose(our_returns, vendor_returns, atol=1e-9)


def test_adjusted_close_has_no_split_jump(catalog: Catalog) -> None:
    snap = catalog.as_of(START + timedelta(days=N_DAYS))
    adj = snap.adjusted_close("TST")
    rets = np.abs(adj[1:] / adj[:-1] - 1.0)
    assert rets[SPLIT_DAY - 1] < 0.05  # no artificial -50% jump at the split


def test_snapshot_is_immutable(catalog: Catalog) -> None:
    snap = catalog.as_of(START + timedelta(days=40))
    with pytest.raises(dataclasses.FrozenInstanceError):
        snap.ts = START  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        snap._data = {}  # type: ignore[misc]
    with pytest.raises(TypeError):
        snap._data["X"] = snap._data["TST"]  # type: ignore[index]
    hist = snap._data["TST"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        hist.close = hist.close  # type: ignore[misc]
    with pytest.raises(ValueError):
        hist.close[0] = 1.0  # read-only numpy array
    with pytest.raises(ValueError):
        snap.adjusted_close("TST")[0] = 1.0


def test_snapshot_hash_changes_with_ts(catalog: Catalog) -> None:
    a = catalog.as_of(START + timedelta(days=40))
    b = catalog.as_of(START + timedelta(days=41))
    assert a.hash() != b.hash()
    assert a.hash() == catalog.as_of(START + timedelta(days=40)).hash()
