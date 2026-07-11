from __future__ import annotations

from decimal import Decimal

import pytest

from twenty.costs.slippage import (
    HalfSpreadCost,
    SquareRootImpact,
    assert_impact_immaterial,
)


def test_half_spread_spy_one_bp() -> None:
    cost = HalfSpreadCost().cost("SPY", Decimal("10000"))
    assert cost == Decimal("0.5000")  # half of 1bp on $10k


def test_half_spread_unknown_symbol_uses_default() -> None:
    assert HalfSpreadCost().cost("XXXX", Decimal("10000")) == Decimal("2.5000")


def test_impact_immaterial_at_our_size() -> None:
    impact = SquareRootImpact().cost_usd(5.0, 20e9)
    assert impact < 0.0001
    assert_impact_immaterial("SPY", 5.0)
    assert_impact_immaterial("EFA", 15.0)


def test_impact_guard_trips_on_absurd_order() -> None:
    with pytest.raises(AssertionError):
        assert_impact_immaterial("EFA", 5e9)
