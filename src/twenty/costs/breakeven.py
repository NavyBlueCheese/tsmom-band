"""Breakeven CLI: what gross return does the account need just to stand still?

    python -m twenty.costs.breakeven --capital 20 --gross-return 0.07 --round-trips 3
"""

from __future__ import annotations

from decimal import Decimal

import typer

from twenty.costs.ibkr import VALUE_CAP_RATE

app = typer.Typer(add_completion=False)

ROUND_TRIP_SCHEDULE = (1, 3, 12, 52)


def annual_drag_fraction(round_trips: int, turnover_fraction: Decimal = Decimal(1)) -> Decimal:
    """Annual commission drag as a fraction of capital, assuming every order
    pays the 1% value cap (always true below $35 notional) and each round
    trip turns over ``turnover_fraction`` of the account."""
    round_trip_cost = 2 * VALUE_CAP_RATE  # buy 1% + sell 1%
    return round_trip_cost * turnover_fraction * round_trips


@app.command()
def main(
    capital: float = typer.Option(20.0, help="Account size in USD"),
    gross_return: float = typer.Option(0.07, help="Assumed annual gross return"),
    round_trips: int = typer.Option(3, help="Full-portfolio round trips per year"),
) -> None:
    cap = Decimal(str(capital))
    gross = Decimal(str(gross_return))
    rt_cost = 2 * VALUE_CAP_RATE

    print(f"Capital: ${cap}")
    print(f"Round-trip cost at this size: {rt_cost:.2%} of position "
        f"(the 1% commission cap binds below $35 notional)")
    print()
    header = f"{'round trips/yr':>14} {'annual drag':>12} {'drag $':>9} {'min gross for net>0':>20}"
    print(header)
    print("-" * len(header))
    schedule = sorted(set(ROUND_TRIP_SCHEDULE) | {round_trips})
    for rt in schedule:
        drag = annual_drag_fraction(rt)
        marker = "  <- requested" if rt == round_trips else ""
        print(
            f"{rt:>14d} {drag:>11.1%} {drag * cap:>8.2f} {drag:>19.1%} {marker}"
        )
    print()
    drag = annual_drag_fraction(round_trips)
    net = gross - drag
    print(f"At {round_trips} round trips/yr: gross {gross:.1%}, drag {drag:.1%}, "
          f"net {net:.1%} (${net * cap:.2f} on ${cap})")
    if net <= 0:
        print("Net is NEGATIVE. The strategy cannot pay this many round trips.")
    print()
    print("Note the last row: weekly rebalancing at this account size costs "
        f"{annual_drag_fraction(52):.0%} of capital per year in commissions alone.")


if __name__ == "__main__":
    app()
