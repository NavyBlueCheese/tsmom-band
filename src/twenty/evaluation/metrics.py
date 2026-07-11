"""Performance metrics. Every function states its annualisation assumption."""

from __future__ import annotations

import math

import numpy as np

TRADING_DAYS = 252


def annualised_return(equity: np.ndarray, periods_per_year: int = TRADING_DAYS) -> float:
    """Geometric annualised return. Assumes ``equity`` is sampled once per
    period and there are ``periods_per_year`` periods per year (default:
    daily bars, 252/year)."""
    if equity.shape[0] < 2 or equity[0] <= 0:
        return 0.0
    total = float(equity[-1] / equity[0])
    years = (equity.shape[0] - 1) / periods_per_year
    if years <= 0 or total <= 0:
        return 0.0
    return float(total ** (1.0 / years)) - 1.0


def annualised_vol(returns: np.ndarray, periods_per_year: int = TRADING_DAYS) -> float:
    """Sample std of per-period returns scaled by sqrt(periods_per_year).
    Assumes i.i.d. periods (see sharpe_lo for the correction when they are
    not)."""
    if returns.shape[0] < 2:
        return 0.0
    return float(np.std(returns, ddof=1)) * math.sqrt(periods_per_year)


def sharpe_naive(returns: np.ndarray, periods_per_year: int = TRADING_DAYS) -> float:
    """mean/std * sqrt(periods_per_year), assuming zero risk-free rate and
    i.i.d. per-period returns. Overstates Sharpe when returns are
    autocorrelated; use sharpe_lo for anything reported."""
    if returns.shape[0] < 2:
        return 0.0
    sd = float(np.std(returns, ddof=1))
    if sd == 0:
        return 0.0
    return float(np.mean(returns)) / sd * math.sqrt(periods_per_year)


def sharpe_lo(
    returns: np.ndarray, periods_per_year: int = TRADING_DAYS, max_lag: int = 10
) -> float:
    """Sharpe with the Lo (2002) correction for serial correlation.

    Annualises by q = periods_per_year using
        SR_q = SR_1 * q / sqrt(q + 2 * sum_{k=1}^{m} (q - k) * rho_k)
    with rho_k the lag-k autocorrelation up to ``max_lag``. For i.i.d.
    returns this reduces to sqrt(q) scaling. Quarterly-rebalanced daily
    returns are autocorrelated, and naive sqrt(252) overstates them.
    """
    n = returns.shape[0]
    if n < max_lag + 2:
        return sharpe_naive(returns, periods_per_year)
    sd = float(np.std(returns, ddof=1))
    if sd == 0:
        return 0.0
    sr1 = float(np.mean(returns)) / sd
    q = periods_per_year
    demeaned = returns - returns.mean()
    denom = float(np.dot(demeaned, demeaned))
    if denom == 0:
        return 0.0
    correction = float(q)
    for k in range(1, max_lag + 1):
        rho_k = float(np.dot(demeaned[:-k], demeaned[k:])) / denom
        correction += 2.0 * (q - k) * rho_k
    if correction <= 0:
        return 0.0
    return sr1 * q / math.sqrt(correction)


def sortino(returns: np.ndarray, periods_per_year: int = TRADING_DAYS) -> float:
    """Mean over downside deviation, scaled by sqrt(periods_per_year),
    zero MAR. Same i.i.d. caveat as sharpe_naive."""
    if returns.shape[0] < 2:
        return 0.0
    downside = returns[returns < 0]
    if downside.shape[0] == 0:
        return math.inf if float(np.mean(returns)) > 0 else 0.0
    dd = float(np.sqrt(np.mean(downside**2)))
    if dd == 0:
        return 0.0
    return float(np.mean(returns)) / dd * math.sqrt(periods_per_year)


def max_drawdown(equity: np.ndarray) -> float:
    """Largest peak-to-trough decline as a positive fraction. No
    annualisation; it is a path statistic."""
    if equity.shape[0] == 0:
        return 0.0
    peaks = np.maximum.accumulate(equity)
    drawdowns = 1.0 - equity / peaks
    return float(drawdowns.max())


def annualised_turnover(
    traded_notional: float, mean_equity: float, n_days: int
) -> float:
    """One-sided traded notional over mean equity, scaled to a 252-day year."""
    if mean_equity <= 0 or n_days <= 0:
        return 0.0
    return traded_notional / mean_equity * (TRADING_DAYS / n_days)


def hit_rate(period_returns: np.ndarray) -> float:
    """Fraction of periods with positive return. Period length is whatever
    the caller passed in; no annualisation."""
    nonzero = period_returns[period_returns != 0]
    if nonzero.shape[0] == 0:
        return 0.0
    return float((nonzero > 0).mean())


def average_holding_period_days(n_leg_trades: int, n_days: int, n_assets: int) -> float:
    """Crude average holding period in days: asset-days divided by leg
    trades. No annualisation."""
    if n_leg_trades == 0:
        return float(n_days)
    return n_days * n_assets / n_leg_trades
