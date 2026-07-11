"""The tearsheet: one HTML file, preregistration first, honesty checks wired
in and not bypassable. Holdout metrics appear only under --unseal."""

from __future__ import annotations

import html as html_mod
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import typer

from twenty.costs.breakeven import ROUND_TRIP_SCHEDULE, annual_drag_fraction
from twenty.evaluation import metrics
from twenty.evaluation.trials import assert_trials_cover_ledgers, distinct_config_count

app = typer.Typer(add_completion=False)

CAPITAL = 20.0


def _ledger_stats(path: Path) -> dict[str, Any]:
    frame = pl.read_parquet(path).sort("ts")
    equity = frame["equity"].cast(pl.Float64).to_numpy()
    returns = np.diff(equity) / equity[:-1]
    fills: list[dict[str, Any]] = []
    for ts, fills_json in zip(
        frame["ts"].to_list(), frame["fills"].to_list(), strict=True
    ):
        for f in json.loads(fills_json):
            f["session"] = ts
            fills.append(f)
    commission = sum(float(f["commission"]) for f in fills)
    slippage = sum(float(f["slippage"]) for f in fills)
    traded = sum(float(f["shares"]) * float(f["price"]) for f in fills)
    return {
        "ts": frame["ts"].to_list(),
        "equity": equity,
        "returns": returns,
        "fills": fills,
        "commission": commission,
        "slippage": slippage,
        "traded": traded,
    }


def _metrics_table(stats: dict[str, Any]) -> str:
    equity = stats["equity"]
    returns = stats["returns"]
    n_days = equity.shape[0]
    rows = [
        ("Annualised return", f"{metrics.annualised_return(equity):.2%}"),
        ("Annualised vol", f"{metrics.annualised_vol(returns):.2%}"),
        ("Sharpe (Lo 2002 corrected)", f"{metrics.sharpe_lo(returns):.2f}"),
        ("Sharpe (naive, for reference)", f"{metrics.sharpe_naive(returns):.2f}"),
        ("Sortino", f"{metrics.sortino(returns):.2f}"),
        ("Max drawdown", f"{metrics.max_drawdown(equity):.2%}"),
        (
            "Annualised turnover",
            f"{metrics.annualised_turnover(stats['traded'], float(np.mean(equity)), n_days):.1%}",
        ),
        ("Leg trades", f"{len(stats['fills'])}"),
        ("Commission paid", f"${stats['commission']:.2f}"),
        ("Spread paid", f"${stats['slippage']:.2f}"),
    ]
    body = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in rows)
    return f"<table><tbody>{body}</tbody></table>"


def _equity_figures(stats: dict[str, Any]) -> str:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    ts = [datetime.fromisoformat(t) for t in stats["ts"]]
    equity = stats["equity"]
    peaks = np.maximum.accumulate(equity)
    drawdown = equity / peaks - 1.0

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, row_heights=[0.5, 0.25, 0.25],
        subplot_titles=("Equity ($)", "Drawdown", "Quarterly returns"),
    )
    fig.add_trace(go.Scatter(x=ts, y=equity, name="equity"), row=1, col=1)
    fig.add_trace(go.Scatter(x=ts, y=drawdown, name="drawdown", fill="tozeroy"), row=2, col=1)

    q_ts: list[datetime] = []
    q_ret: list[float] = []
    last_idx = 0
    for i in range(1, len(ts)):
        if ts[i].month != ts[i - 1].month and ts[i].month in (1, 4, 7, 10):
            q_ts.append(ts[i - 1])
            q_ret.append(float(equity[i - 1] / equity[last_idx] - 1.0))
            last_idx = i
    fig.add_trace(go.Bar(x=q_ts, y=q_ret, name="quarterly return"), row=3, col=1)
    fig.update_layout(height=800, showlegend=False)
    return str(fig.to_html(full_html=False, include_plotlyjs="cdn"))


def _waterfall(stats: dict[str, Any]) -> str:
    import plotly.graph_objects as go

    net = float(stats["equity"][-1]) - CAPITAL
    gross = net + stats["commission"] + stats["slippage"]
    fig = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=["absolute", "relative", "relative", "total"],
            x=["Gross PnL", "Commission", "Spread", "Net PnL"],
            y=[gross, -stats["commission"], -stats["slippage"], net],
            text=[f"${v:+.2f}" for v in (gross, -stats["commission"], -stats["slippage"], net)],
        )
    )
    fig.update_layout(title="Cost attribution ($, whole period)", height=400)
    return str(fig.to_html(full_html=False, include_plotlyjs=False))


def _trade_log(stats: dict[str, Any]) -> str:
    rows = []
    for f in stats["fills"]:
        notional = float(f["shares"]) * float(f["price"])
        rows.append(
            f"<tr><td>{f['session'][:10]}</td><td>{f['symbol']}</td>"
            f"<td>{f['side']}</td><td>${notional:.2f}</td>"
            f"<td>${float(f['commission']):.2f}</td></tr>"
        )
    return (
        "<table><thead><tr><th>Date</th><th>Symbol</th><th>Side</th>"
        "<th>Notional</th><th>Commission</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _breakeven_table() -> str:
    rows = []
    for rt in ROUND_TRIP_SCHEDULE:
        drag = float(annual_drag_fraction(rt))
        rows.append(
            f"<tr><td>{rt}</td><td>{drag:.1%}</td><td>${drag * CAPITAL:.2f}</td></tr>"
        )
    return (
        "<table><thead><tr><th>Round trips/yr</th><th>Annual drag</th>"
        "<th>Drag on $20</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


@app.command()
def main(
    ledger: Path = typer.Option(Path("ledgers/band_0.20.parquet")),
    holdout_ledger: Path = typer.Option(Path("ledgers/holdout_band_0.20.parquet")),
    out: Path = typer.Option(Path("research/tearsheet.html")),
    unseal: bool = typer.Option(False, help="Include holdout metrics. Irreversible in spirit."),
) -> None:
    import markdown as md

    # The honesty mechanism. Not bypassable: no flag skips this.
    n_trials = assert_trials_cover_ledgers(Path("ledgers"))
    n_configs = distinct_config_count()

    prereg_html = md.markdown(Path("PREREGISTRATION.md").read_text(encoding="utf-8"))
    stats = _ledger_stats(ledger)

    sections = [
        "<h1>twenty: tearsheet</h1>",
        f"<p>Generated {datetime.now().isoformat(timespec='seconds')}</p>",
        "<h2>Preregistration</h2>",
        f"<blockquote>{prereg_html}</blockquote>",
        "<h2>Trial count</h2>",
        f"<p>{n_trials} trials logged in trials.jsonl "
        f"({n_configs} distinct configurations). Every reported number below "
        "must be deflated against that count.</p>",
        "<h2>Breakeven (cost floor before any strategy)</h2>",
        _breakeven_table(),
        "<h2>In-sample metrics (train 2005-2018 only)</h2>",
        _metrics_table(stats),
        "<h2>Equity, drawdown, quarterly returns</h2>",
        _equity_figures(stats),
        "<h2>Cost attribution</h2>",
        _waterfall(stats),
        "<h2>Trade log (every leg-trade, whole train period)</h2>",
        _trade_log(stats),
    ]

    if unseal:
        print("WARNING: unsealing the holdout. This is a one-time read; you do "
              "not get another untouched holdout.")
        if holdout_ledger.exists():
            h_stats = _ledger_stats(holdout_ledger)
            sections += [
                "<h2>HOLDOUT metrics (2019-)</h2>",
                _metrics_table(h_stats),
                _equity_figures(h_stats),
            ]
        else:
            sections += [
                "<h2>HOLDOUT</h2>",
                f"<p>No holdout ledger at {html_mod.escape(str(holdout_ledger))}. "
                "Run the backtest over data/holdout first.</p>",
            ]
    else:
        sections += [
            "<h2>Holdout</h2>",
            "<p>Sealed. Pass --unseal to include it, once, at the end.</p>",
        ]

    style = (
        "<style>body{font-family:Georgia,serif;max-width:1000px;margin:2rem auto;"
        "padding:0 1rem;line-height:1.5}table{border-collapse:collapse}"
        "td,th{border:1px solid #999;padding:4px 10px;text-align:right}"
        "th{background:#eee}blockquote{background:#f6f6f6;padding:1rem;"
        "border-left:4px solid #888}</style>"
    )
    out.parent.mkdir(exist_ok=True)
    out.write_text(
        f"<!doctype html><html><head><meta charset='utf-8'>{style}</head>"
        f"<body>{''.join(sections)}</body></html>",
        encoding="utf-8",
    )
    print(f"Wrote {out}")
    if not math.isfinite(float(stats["equity"][-1])):
        raise RuntimeError("Ledger equity is not finite; refusing to report")


if __name__ == "__main__":
    app()
