"""Spread and impact models.

Spread cost is half the quoted spread. Market impact is zero at $5 notional;
SquareRootImpact exists as a guard — if it ever returns something material,
the position sizing has a bug.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal

# Assumed full quoted spreads, in basis points, configurable per instance.
DEFAULT_SPREAD_BPS: dict[str, int] = {"SPY": 1, "IEF": 3, "GLD": 3, "EFA": 4}


@dataclass(frozen=True)
class HalfSpreadCost:
    """Cost of crossing half the quoted spread on a marketable order."""

    spread_bps: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_SPREAD_BPS))
    default_bps: int = 5

    def cost(self, symbol: str, notional: Decimal) -> Decimal:
        bps = self.spread_bps.get(symbol, self.default_bps)
        half = Decimal(bps) / Decimal(2) / Decimal(10_000)
        return (abs(notional) * half).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)


@dataclass(frozen=True)
class SquareRootImpact:
    """Square-root market impact (Almgren et al. 2005):

        impact_fraction = eta * sigma_daily * sqrt(size / ADV)

    At $5 orders against multi-billion-dollar ADV this is sub-hundredth-of-a-
    cent. It is implemented as a tripwire, not a cost: the engine asserts it
    stays below $0.0001 and treats anything larger as evidence of a sizing bug.
    """

    eta: float = 0.1
    sigma_daily: float = 0.01

    def cost_usd(self, notional_usd: float, adv_usd: float) -> float:
        if adv_usd <= 0:
            raise ValueError("ADV must be positive")
        size_fraction = abs(notional_usd) / adv_usd
        impact_fraction = self.eta * self.sigma_daily * math.sqrt(size_fraction)
        return abs(notional_usd) * impact_fraction


# Conservative (low) average daily dollar volumes; real figures are higher,
# which would make impact even smaller.
APPROX_ADV_USD: dict[str, float] = {
    "SPY": 20e9,
    "EFA": 1e9,
    "IEF": 1e9,
    "GLD": 2e9,
}
IMPACT_GUARD_USD = 0.0001


def assert_impact_immaterial(symbol: str, notional_usd: float) -> None:
    """Raise if square-root impact on this order is material. It never should
    be at $20 of capital; if it is, the sizing produced an absurd order."""
    impact = SquareRootImpact().cost_usd(notional_usd, APPROX_ADV_USD.get(symbol, 1e9))
    if impact >= IMPACT_GUARD_USD:
        raise AssertionError(
            f"Market impact {impact:.6f} USD on {symbol} order of {notional_usd:.2f} USD "
            "is material; position sizing has a bug"
        )
