"""Probabilistic and Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014).

The CLI requires --n-trials with no default, and cross-checks it against
trials.jsonl: passing fewer trials than were actually run is refused.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import typer
from scipy import stats

from twenty.evaluation.trials import distinct_config_count

app = typer.Typer(add_completion=False)

EULER_MASCHERONI = 0.5772156649015329


def probabilistic_sharpe(
    observed_sr: float,
    benchmark_sr: float,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """P(true SR > benchmark_sr) given an observed per-period SR over n_obs
    periods, with the Mertens (2002) standard error that adjusts for skew and
    excess kurtosis. All SRs are per-period (not annualised)."""
    if n_obs < 2:
        return 0.0
    variance = (
        1.0
        - skew * observed_sr
        + (kurtosis - 1.0) / 4.0 * observed_sr**2
    ) / (n_obs - 1)
    if variance <= 0:
        return 0.0
    z = (observed_sr - benchmark_sr) / math.sqrt(variance)
    return float(stats.norm.cdf(z))


def expected_max_sharpe(n_trials: int, var_trials_sr: float, n_obs: int) -> float:
    """E[max SR] across n_trials of zero-true-SR strategies (the false
    discovery benchmark), per Bailey & Lopez de Prado 2014."""
    if n_trials <= 1:
        return 0.0
    sd = math.sqrt(var_trials_sr) if var_trials_sr > 0 else 1.0 / math.sqrt(n_obs)
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return float(sd * ((1.0 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2))


def deflated_sharpe(
    returns: np.ndarray, n_trials: int, var_trials_sr: float | None = None
) -> tuple[float, float, float]:
    """(observed per-period SR, benchmark SR0, DSR). DSR is the PSR evaluated
    against the expected max SR of ``n_trials`` skill-less strategies."""
    n = returns.shape[0]
    sd = float(np.std(returns, ddof=1))
    sr = float(np.mean(returns)) / sd if sd > 0 else 0.0
    skew = float(stats.skew(returns))
    kurt = float(stats.kurtosis(returns, fisher=False))
    var_sr = var_trials_sr if var_trials_sr is not None else 1.0 / n
    sr0 = expected_max_sharpe(n_trials, var_sr, n)
    return sr, sr0, probabilistic_sharpe(sr, sr0, n, skew, kurt)


@app.command()
def main(
    n_trials: int = typer.Option(..., help="Number of configurations ever tried"),
    ledger: Path = typer.Option(
        Path("ledgers/band_0.20.parquet"), help="Ledger parquet to evaluate"
    ),
    trials_file: Path = typer.Option(Path("trials.jsonl")),
) -> None:
    import polars as pl

    recorded = distinct_config_count(trials_file)
    if n_trials < recorded:
        raise typer.BadParameter(
            f"--n-trials {n_trials} is below the {recorded} distinct configs "
            f"recorded in {trials_file}. You do not get to forget trials; refusing."
        )
    frame = pl.read_parquet(ledger)
    equity = frame["equity"].cast(pl.Float64).to_numpy()
    returns = np.diff(equity) / equity[:-1]
    sr, sr0, dsr = deflated_sharpe(returns, n_trials)
    ann = sr * math.sqrt(252.0)
    print(f"Observed daily SR: {sr:.4f} (naive annualised {ann:.2f})")
    print(f"Expected max daily SR of {n_trials} skill-less trials: {sr0:.4f}")
    print(f"Deflated Sharpe Ratio (P(true SR > that benchmark)): {dsr:.1%}")


if __name__ == "__main__":
    app()
