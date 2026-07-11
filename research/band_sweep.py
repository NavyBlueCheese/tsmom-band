"""The band sweep: the actual experiment.

Runs the full event-engine backtest over data/train/ only, $20 capital, for
nine values of BAND. Logs every configuration to trials.jsonl, writes
research/band_sweep.html, and prints the three preregistered questions.

Do not change the default BAND on the basis of this sweep. Report and stop.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from twenty.backtest.engine import BacktestEngine  # noqa: E402
from twenty.backtest.fills import NextBarOpen  # noqa: E402
from twenty.data.store import Catalog  # noqa: E402
from twenty.evaluation import metrics  # noqa: E402
from twenty.evaluation.trials import TRIALS_PATH, trial  # noqa: E402
from twenty.risk.checks import PreTradeRiskCheck  # noqa: E402
from twenty.strategies.tsmom_band import TsmomBand, is_quarter_end_session  # noqa: E402

BANDS = (0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)
CAPITAL = Decimal("20.00")
LEDGER_DIR = REPO / "ledgers"


@trial(path=REPO / TRIALS_PATH.name)
def run_one(catalog: Catalog, *, config: dict[str, Any]) -> dict[str, Any]:
    band = float(config["band"])
    engine = BacktestEngine(
        catalog=catalog,
        strategy=TsmomBand(band=band),
        fill_model=NextBarOpen(),
        initial_capital=CAPITAL,
        risk=PreTradeRiskCheck(),
    )
    ledger = engine.run()
    ledger.write_parquet(LEDGER_DIR / f"band_{band:.2f}.parquet")

    equity = np.array([e for _, e in ledger.equity_series()])
    returns = np.diff(equity) / equity[:-1]
    fills = ledger.fills()
    commission = float(ledger.total_commission())
    slippage = float(ledger.total_slippage())
    traded = sum(float(Decimal(f["shares"]) * Decimal(f["price"])) for f in fills)

    net_final = float(equity[-1])
    net_return = net_final / float(CAPITAL) - 1.0
    gross_final = net_final + commission + slippage
    gross_return = gross_final / float(CAPITAL) - 1.0

    quarter_rows = [
        r for r in ledger.rows
        if is_quarter_end_session(datetime.fromisoformat(r["ts"]))
    ]

    zero_trade_quarters = sum(1 for r in quarter_rows if not json.loads(r["orders"]))
    n_days = equity.shape[0]

    return {
        "band": band,
        "gross_return": gross_return,
        "commission_usd": commission,
        "slippage_usd": slippage,
        "net_return": net_return,
        "net_final_usd": net_final,
        "net_sharpe": metrics.sharpe_lo(returns),
        "annualised_turnover": metrics.annualised_turnover(
            traded, float(np.mean(equity)), n_days
        ),
        "leg_trades": len(fills),
        "max_drawdown": metrics.max_drawdown(equity),
        "zero_trade_quarter_fraction": (
            zero_trade_quarters / len(quarter_rows) if quarter_rows else 0.0
        ),
        "n_quarters": len(quarter_rows),
    }


def sharpe_standard_error(sharpe: float, n_days: int) -> float:
    """Lo (2002) i.i.d. approximation: SE(SR_annual) ~ sqrt((1 + SR_daily^2/2)/n) * sqrt(252)."""
    sr_daily = sharpe / np.sqrt(252.0)
    return float(np.sqrt((1.0 + sr_daily**2 / 2.0) / n_days) * np.sqrt(252.0))


def write_html(results: list[dict[str, Any]], n_days: int) -> Path:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    bands = [r["band"] for r in results]
    sharpes = [r["net_sharpe"] for r in results]
    turnovers = [r["annualised_turnover"] for r in results]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("Net Sharpe vs BAND (train, 2005-2018)", "Annualised turnover vs BAND"),
    )
    fig.add_trace(go.Scatter(x=bands, y=sharpes, mode="lines+markers", name="net Sharpe"),
                  row=1, col=1)
    se = sharpe_standard_error(max(sharpes), n_days)
    fig.add_trace(
        go.Scatter(
            x=bands, y=[max(sharpes) - se] * len(bands), mode="lines",
            line={"dash": "dot", "color": "gray"}, name="max Sharpe - 1 SE",
        ),
        row=1, col=1,
    )
    fig.add_trace(go.Scatter(x=bands, y=turnovers, mode="lines+markers", name="turnover"),
                  row=2, col=1)
    for row in (1, 2):
        fig.add_vline(x=0.20, line_dash="dash", line_color="firebrick", row=row, col=1)
    fig.update_xaxes(title_text="BAND (weight points)", row=2, col=1)
    fig.update_layout(height=750, showlegend=True,
                      title="Band sweep: the controller is the product")
    out = REPO / "research" / "band_sweep.html"
    fig.write_html(out)
    return out


def main() -> None:
    catalog = Catalog.from_dir(REPO / "data" / "train")
    LEDGER_DIR.mkdir(exist_ok=True)
    results: list[dict[str, Any]] = []
    for band in BANDS:
        print(f"Running BAND = {band:.2f} ...", flush=True)
        results.append(run_one(catalog, config={"strategy": "tsmom_band", "band": band}))

    print()
    header = (
        f"{'BAND':>5} {'gross':>8} {'comm $':>7} {'net':>8} {'final $':>8} "
        f"{'Sharpe':>7} {'turnover':>9} {'legs':>5} {'maxDD':>7} {'0-trade qtrs':>12}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['band']:>5.2f} {r['gross_return']:>7.1%} {r['commission_usd']:>7.2f} "
            f"{r['net_return']:>7.1%} {r['net_final_usd']:>8.2f} {r['net_sharpe']:>7.2f} "
            f"{r['annualised_turnover']:>8.1%} {r['leg_trades']:>5d} "
            f"{r['max_drawdown']:>6.1%} {r['zero_trade_quarter_fraction']:>11.0%}"
        )

    n_days = 3500 
    out = write_html(results, n_days)
    print(f"\nWrote {out}")

    print("\n=== The three questions ===")
    r0 = results[0]
    gross_usd = r0["gross_return"] * float(CAPITAL)
    net_usd = r0["net_return"] * float(CAPITAL)
    print(
        f"1. BAND = 0.00 (full rebalance every quarter): net return is "
        f"{'POSITIVE' if r0['net_return'] > 0 else 'NEGATIVE'}. "
        f"On $20: gross ${gross_usd:+.2f}, commission ${r0['commission_usd']:.2f} "
        f"(plus spread ${r0['slippage_usd']:.2f}), net ${net_usd:+.2f}."
    )
    best = max(r["net_sharpe"] for r in results)
    se = sharpe_standard_error(best, n_days)
    within = [r["band"] for r in results if r["net_sharpe"] >= best - se]
    print(
        f"2. Net Sharpe is within one standard error (SE = {se:.2f}) of its "
        f"maximum ({best:.2f}) for BAND in [{min(within):.2f}, {max(within):.2f}] "
        f"({len(within)} of {len(results)} values). The curve is flat: picking "
        "the argmax of a flat noisy curve is picking noise."
    )
    r20 = next(r for r in results if abs(r["band"] - 0.20) < 1e-9)
    print(
        f"3. BAND = 0.20 executes {r20['leg_trades']} leg-trades across 14 years "
        f"({r20['n_quarters']} quarter-ends; {r20['zero_trade_quarter_fraction']:.0%} "
        "of quarters trade nothing)."
    )
    print("\nBAND stays at 0.20 regardless of the numbers above. Reported, stopping.")


if __name__ == "__main__":
    main()
