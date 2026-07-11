"""The point of the whole exercise: a falsifiable prediction.

Block-bootstraps the train-period backtest into a distribution of one-year
net returns and writes PREDICTION.md. Commit it before funding the account;
in twelve months, compare. That comparison — did reality land inside the
predicted interval — is the only result from this project that means
anything.
"""

from __future__ import annotations

import json
import subprocess
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import typer

app = typer.Typer(add_completion=False)

CAPITAL = 20.0
BLOCK_QUARTERS = 4
N_DRAWS = 10_000


def quarterly_returns_and_trades(ledger_path: Path) -> tuple[np.ndarray, float]:
    frame = pl.read_parquet(ledger_path).sort("ts")
    equity = frame["equity"].cast(pl.Float64).to_numpy()
    ts = frame["ts"].to_list()
    fills_per_row = [len(json.loads(f)) for f in frame["fills"].to_list()]
    # Quarter boundaries: month changes into {4,7,10,1}.
    boundaries: list[int] = []
    for i in range(1, len(ts)):
        prev_month = ts[i - 1][5:7]
        month = ts[i][5:7]
        if month != prev_month and month in ("01", "04", "07", "10"):
            boundaries.append(i)
    starts = [0, *boundaries]
    ends = [*boundaries, len(ts)]
    rets: list[float] = []
    trades: list[int] = []
    for start, end in zip(starts, ends, strict=True):
        if end - start < 5:
            continue
        rets.append(equity[end - 1] / equity[start] - 1.0)
        trades.append(sum(fills_per_row[start:end]))
    return np.array(rets), float(np.mean(trades)) if trades else 0.0


def bootstrap_one_year(
    quarterly: np.ndarray, n_draws: int = N_DRAWS, seed: int = 20
) -> np.ndarray:
    """Block bootstrap with 4-quarter blocks: each draw picks one contiguous
    block of four quarters, preserving intra-year autocorrelation."""
    rng = np.random.default_rng(seed)
    n = quarterly.shape[0]
    max_start = n - BLOCK_QUARTERS
    if max_start < 1:
        raise ValueError("Not enough quarters for 4-quarter blocks")
    starts = rng.integers(0, max_start + 1, size=n_draws)
    out = np.empty(n_draws)
    for i, s in enumerate(starts):
        out[i] = float(np.prod(1.0 + quarterly[s : s + BLOCK_QUARTERS]) - 1.0)
    return out


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "(not committed yet)"


@app.command()
def main(
    ledger: Path = typer.Option(Path("ledgers/band_0.20.parquet")),
    out: Path = typer.Option(Path("PREDICTION.md")),
) -> None:
    quarterly, mean_trades_per_quarter = quarterly_returns_and_trades(ledger)
    draws = bootstrap_one_year(quarterly)
    pnl = draws * CAPITAL

    expected = float(np.mean(pnl))
    median = float(np.median(pnl))
    sd = float(np.std(pnl, ddof=1))
    p10, p90 = (float(np.percentile(pnl, q)) for q in (10, 90))
    p_up = float((pnl > 0).mean())
    trades_year = mean_trades_per_quarter * 4

    lines = [
        "# Prediction",
        "",
        f"Date: {date.today().isoformat()}",
        f"Git SHA: {git_sha()}",
        f"Source: block bootstrap ({BLOCK_QUARTERS}-quarter blocks, "
        f"{N_DRAWS:,} draws) of the train-period backtest at BAND = 0.20, "
        f"on ${CAPITAL:.0f}.",
        "",
        "| Quantity | Value |",
        "|---|---|",
        f"| Expected one-year PnL | ${expected:+.2f} |",
        f"| Median one-year PnL | ${median:+.2f} |",
        f"| Standard deviation | ${sd:.2f} |",
        f"| 10th percentile | ${p10:+.2f} |",
        f"| 90th percentile | ${p90:+.2f} |",
        f"| P(account is up after one year) | {p_up:.0%} |",
        f"| Expected leg-trades per year | {trades_year:.1f} |",
        "",
        "Success criterion (preregistered): live PnL after one year lies inside "
        f"the 80% interval [${p10:+.2f}, ${p90:+.2f}]. Not: live PnL is positive.",
        "",
        "Commit this file before funding the account. In twelve months, compare.",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    app()
