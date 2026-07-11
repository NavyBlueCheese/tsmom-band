"""Golden tests against IBKR's published worked examples, plus the structural
fact the project rests on: below $35 notional the 1% cap binds exactly."""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from twenty.costs.ibkr import commission, order_cost, sell_regulatory_fees


def test_golden_half_share_at_ten_dollars() -> None:
    # IBKR worked example: 0.5 shares at $10.00, notional $5.00 -> $0.05
    assert commission(Decimal("0.5"), Decimal("10.00")) == Decimal("0.05")


def test_golden_five_hundredths_share_at_fifteen() -> None:
    # IBKR worked example: 0.05 shares at $15.00, notional $0.75 -> $0.01 floor
    assert commission(Decimal("0.05"), Decimal("15.00")) == Decimal("0.01")


def test_minimum_binds_for_small_whole_share_orders() -> None:
    # 10 shares at $50: per-share 0.035 < minimum 0.35; 1% cap = 5.00 -> 0.35
    assert commission(Decimal("10"), Decimal("50.00")) == Decimal("0.35")


@given(
    notional_cents=st.integers(min_value=200, max_value=3499),
    price_cents=st.integers(min_value=100, max_value=100_000),
)
@settings(max_examples=300, deadline=None)
def test_one_percent_cap_binds_below_35_dollars(notional_cents: int, price_cents: int) -> None:
    """For any notional in [2, 34.99], commission / notional == 1% exactly
    (up to the cent rounding of the invoice). Shares are quantized to IBKR's
    0.0001 fractional resolution, as a real order would be."""
    requested = Decimal(notional_cents) / 100
    price = Decimal(price_cents) / 100
    shares = (requested / price).quantize(Decimal("0.0001"))
    assume(shares > 0)
    actual_notional = shares * price
    assume(Decimal("2") <= actual_notional < Decimal("35"))
    c = commission(shares, price)
    expected = (Decimal("0.01") * actual_notional).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )
    expected = max(expected, Decimal("0.01"))
    assert c == expected


def test_sell_fees_positive_and_small() -> None:
    fees = sell_regulatory_fees(Decimal("0.5"), Decimal("10.00"))
    assert Decimal("0") <= fees <= Decimal("0.01")


def test_order_cost_sell_at_least_buy() -> None:
    shares, price = Decimal("100"), Decimal("100.00")
    assert order_cost(shares, price, is_sell=True) >= order_cost(shares, price, is_sell=False)


def test_negative_inputs_raise() -> None:
    with pytest.raises(ValueError):
        commission(Decimal("-1"), Decimal("10"))
