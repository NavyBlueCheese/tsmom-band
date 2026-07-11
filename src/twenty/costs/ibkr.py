"""IBKR Pro Tiered commission model for US stocks and ETFs.

The structural fact the whole project rests on: for any notional below $35,
the 1%-of-trade-value cap binds, so commission / notional == 1% exactly.
A round trip therefore costs 2% of the position, always, at our size.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

PER_SHARE_RATE = Decimal("0.0035")
ORDER_MINIMUM = Decimal("0.35")
VALUE_CAP_RATE = Decimal("0.01")  # commission may not exceed 1% of trade value
FRACTIONAL_FLOOR = Decimal("0.01")

# Regulatory fees charged on sells only.
# Rates looked up 2026-07-10 from
# https://www.interactivebrokers.com/en/pricing/commissions-stocks.php
# (SEC Section 31 fee and FINRA Trading Activity Fee). They change, sometimes
# more than once a year; if you are reading this long after that date,
# re-check before trusting a backtest to the fourth decimal.
REGULATORY_SELL_FEES: dict[str, Decimal] = {
    "sec_section_31_per_dollar": Decimal("0.0000278"),  # per $ of sale proceeds
    "finra_taf_per_share": Decimal("0.000166"),  # per share sold, cap $8.30/trade
}
FINRA_TAF_CAP = Decimal("8.30")

_CENT = Decimal("0.01")


def commission(shares: Decimal, price: Decimal) -> Decimal:
    """Tiered commission for a single order, rounded to the cent."""
    if shares < 0 or price < 0:
        raise ValueError("shares and price must be non-negative")
    notional = shares * price
    raw = max(
        FRACTIONAL_FLOOR,
        min(VALUE_CAP_RATE * notional, max(ORDER_MINIMUM, PER_SHARE_RATE * shares)),
    )
    return raw.quantize(_CENT, rounding=ROUND_HALF_UP)


def sell_regulatory_fees(shares: Decimal, price: Decimal) -> Decimal:
    """SEC Section 31 fee plus FINRA TAF, charged on sells only."""
    if shares < 0 or price < 0:
        raise ValueError("shares and price must be non-negative")
    proceeds = shares * price
    sec_fee = REGULATORY_SELL_FEES["sec_section_31_per_dollar"] * proceeds
    taf = min(REGULATORY_SELL_FEES["finra_taf_per_share"] * shares, FINRA_TAF_CAP)
    return (sec_fee + taf).quantize(_CENT, rounding=ROUND_HALF_UP)


def order_cost(shares: Decimal, price: Decimal, is_sell: bool) -> Decimal:
    """Total broker-side cost of one order: commission plus, on sells,
    regulatory fees."""
    total = commission(shares, price)
    if is_sell:
        total += sell_regulatory_fees(shares, price)
    return total
