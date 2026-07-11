"""Time-series momentum, long/flat, with the no-trade band.

Parameters below are frozen in PREREGISTRATION.md. Do not adjust them, do not
add a tuned variant.
"""

from __future__ import annotations

import functools
from datetime import datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

import numpy as np
import structlog

from twenty.backtest.types import PortfolioView, ProposedOrder, Side
from twenty.data.store import Snapshot

log = structlog.get_logger(__name__)

UNIVERSE = ("SPY", "EFA", "IEF", "GLD")
LOOKBACK = 252  # trading days
SKIP = 21  # trading days
VOL_WINDOW = 60  # trading days
VOL_TARGET = 0.15  # annualised
MAX_LEVERAGE = 1.0
BAND = 0.20  # weight points
MIN_ORDER_USD = Decimal("2.00")
LIMIT_OFFSET_BPS = 3

_MIN_HISTORY = LOOKBACK + SKIP + 1


@functools.lru_cache(maxsize=8)
def _quarter_end_sessions(year: int) -> frozenset[str]:
    """Dates (ISO strings) of the last XNYS session of each calendar quarter
    in ``year``. Computed from the exchange calendar, never date arithmetic."""
    import exchange_calendars as xcals

    # The default calendar only extends about a year past today; pin explicit
    # bounds so any year in the data (or the future schedule) resolves.
    cal = xcals.get_calendar("XNYS", start=f"{year - 1}-12-01", end=f"{year + 1}-01-31")
    out: set[str] = set()
    for month in (3, 6, 9, 12):
        # Last session at or before the last calendar day of the quarter.
        last_day = {3: f"{year}-03-31", 6: f"{year}-06-30",
                    9: f"{year}-09-30", 12: f"{year}-12-31"}[month]
        session = cal.date_to_session(last_day, direction="previous")
        out.add(str(session.date()))
    return frozenset(out)


def is_quarter_end_session(ts: datetime) -> bool:
    """True if ``ts`` falls on the last XNYS session of a calendar quarter.

    Convention: daily bars carry the session date as UTC midnight, and any
    live timestamp during regular trading hours (e.g. the 15:45
    America/New_York trigger) shares its UTC calendar date with the New York
    session date, because New York is behind UTC and its date rolls later.
    The UTC date is therefore the session label in both cases.
    """
    session_date = ts.astimezone(ZoneInfo("UTC")).date()
    return str(session_date) in _quarter_end_sessions(session_date.year)


def momentum(snapshot: Snapshot) -> dict[str, float]:
    """12-month momentum ending one month ago: P[t-SKIP] / P[t-252] - 1.

    With P[-1] being bar t, P[-1 - SKIP] is bar t-21 and
    P[-1 - SKIP - LOOKBACK + 21] is bar t-252.
    """
    out: dict[str, float] = {}
    for symbol in UNIVERSE:
        prices = snapshot.adjusted_close(symbol)
        out[symbol] = float(
            prices[-1 - SKIP] / prices[-1 - SKIP - LOOKBACK + 21] - 1.0
        )
    return out


def signal(snapshot: Snapshot) -> dict[str, int]:
    return {s: 1 if m > 0 else 0 for s, m in momentum(snapshot).items()}


def target_weights(snapshot: Snapshot) -> dict[str, Decimal]:
    """Inverse-vol weights on the long legs, scaled to the vol target and
    capped at 1.0 gross.

    The min() cap is load-bearing. In calm markets the vol target asks for
    roughly 150% gross exposure and a cash account cannot supply it. The
    strategy therefore runs below target most of the time — a real, permanent
    haircut to expected return relative to the published academic results.
    Do not remove the cap to make a backtest look better.
    """
    from sklearn.covariance import LedoitWolf

    sig = signal(snapshot)
    log_returns: dict[str, np.ndarray] = {}
    for symbol in UNIVERSE:
        prices = snapshot.adjusted_close(symbol)
        window = prices[-(VOL_WINDOW + 1):]
        log_returns[symbol] = np.diff(np.log(window))

    w_raw: dict[str, float] = {}
    for symbol in UNIVERSE:
        sigma = float(np.std(log_returns[symbol], ddof=1)) * float(np.sqrt(252.0))
        w_raw[symbol] = (sig[symbol] / sigma) if sigma > 0 else 0.0

    total = sum(w_raw.values())
    if total == 0:
        return {s: Decimal(0) for s in UNIVERSE}
    w_norm = np.array([w_raw[s] / total for s in UNIVERSE])

    returns_matrix = np.column_stack([log_returns[s] for s in UNIVERSE])
    cov = LedoitWolf().fit(returns_matrix).covariance_ * 252.0
    sigma_p = float(np.sqrt(w_norm @ cov @ w_norm))
    k = min(MAX_LEVERAGE, VOL_TARGET / sigma_p) if sigma_p > 0 else 0.0
    return {
        s: Decimal(str(k * float(w_norm[i]))).quantize(Decimal("0.0001"))
        for i, s in enumerate(UNIVERSE)
    }


class TsmomBand:
    """The strategy. Emits orders only on quarter-end sessions, only for legs
    outside the band, sells before buys."""

    def __init__(self, band: float = BAND) -> None:
        # ``band`` is parameterised ONLY so research/band_sweep.py can run the
        # preregistered sweep. Reported results and live trading use BAND.
        self.band = band
        self.state: dict[str, object] = {}

    def on_bar(self, snapshot: Snapshot, portfolio: PortfolioView) -> list[ProposedOrder]:
        if not is_quarter_end_session(snapshot.ts):
            return []

        history = min(snapshot.sessions(s) for s in UNIVERSE)
        if history < _MIN_HISTORY:
            log.warning(
                "Insufficient history, standing aside",
                have=history,
                need=_MIN_HISTORY,
            )
            return []

        marks: dict[str, Decimal] = {}
        for symbol in UNIVERSE:
            closes = snapshot.raw_close(symbol)
            marks[symbol] = Decimal(str(float(closes[-1])))

        w_target = target_weights(snapshot)
        w_current_all = portfolio.weights(marks)
        w_current = {s: w_current_all.get(s, Decimal(0)) for s in UNIVERSE}
        total_value = portfolio.total_value(marks)
        band = Decimal(str(self.band))

        sells: list[ProposedOrder] = []
        buys: list[ProposedOrder] = []
        for symbol in UNIVERSE:
            delta_w = w_target[symbol] - w_current[symbol]
            if abs(delta_w) <= band:
                continue
            delta_notional = delta_w * total_value
            if abs(delta_notional) < MIN_ORDER_USD:
                continue
            order = ProposedOrder(
                symbol=symbol,
                side=Side.BUY if delta_notional > 0 else Side.SELL,
                notional=abs(delta_notional).quantize(Decimal("0.01")),
                limit_offset_bps=LIMIT_OFFSET_BPS,
            )
            # All sells first: there is no margin, and buying before selling
            # produces an insufficient-funds rejection.
            (buys if order.side is Side.BUY else sells).append(order)
        return sells + buys
