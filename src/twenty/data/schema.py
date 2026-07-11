"""Point-in-time bar schema.

Raw prices plus a cumulative adjustment factor, never a pre-adjusted close.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, field_validator


class Bar(BaseModel):
    """One daily bar of raw market data plus point-in-time adjustment state.

    Why raw close + adj_factor instead of a vendor "adjusted close" column:
    a vendor recomputes its entire adjusted-close series every time a split or
    dividend occurs. A series downloaded today therefore embeds knowledge of
    every corporate action up to today — including ones that happened after
    the bar's own date. Feeding that column to a backtest is lookahead.
    Storing the raw close together with the cumulative split/dividend
    adjustment factor *as of that bar's date* lets us reconstruct an adjusted
    series using only information that existed at the time, by rebasing the
    factor to the last bar visible in the snapshot.

    ``dividend`` is the cash distribution per share going ex on this bar's
    date (0.0 otherwise). ``split`` is the share multiplier effective this
    date (1.0 otherwise). Both are point-in-time events on their own timeline;
    the backtest credits dividends to cash and multiplies share counts on the
    ex-date. ``adj_factor`` is the cumulative product view of the same events,
    used only to reconstruct adjusted price series for signals.
    """

    model_config = ConfigDict(frozen=True)

    symbol: str
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    adj_factor: float
    dividend: float = 0.0
    split: float = 1.0

    @field_validator("ts")
    @classmethod
    def _tz_aware_utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None or v.utcoffset() is None:
            raise ValueError("ts must be timezone-aware")
        if v.utcoffset().total_seconds() != 0:  # type: ignore[union-attr]
            raise ValueError("ts must be UTC")
        return v


BAR_COLUMNS: tuple[str, ...] = (
    "symbol",
    "ts",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "adj_factor",
    "dividend",
    "split",
)
