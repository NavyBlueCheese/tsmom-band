"""Data quality checks. Flags problems, never silently drops rows."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import polars as pl
import structlog

log = structlog.get_logger(__name__)

MAX_GAP_SESSIONS = 5
MAX_ABS_RETURN = 0.25


@dataclass
class QualityReport:
    zero_volume: list[tuple[str, str]] = field(default_factory=list)
    nonpositive_close: list[tuple[str, str]] = field(default_factory=list)
    gaps: list[tuple[str, str, int]] = field(default_factory=list)
    extreme_returns: list[tuple[str, str, float]] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not (
            self.zero_volume or self.nonpositive_close or self.gaps or self.extreme_returns
        )

    def summary(self) -> str:
        lines = [
            f"Zero-volume bars: {len(self.zero_volume)}",
            f"Non-positive closes: {len(self.nonpositive_close)}",
            f"Gaps over {MAX_GAP_SESSIONS} sessions: {len(self.gaps)}",
            f"Unexplained |return| > {MAX_ABS_RETURN:.0%}: {len(self.extreme_returns)}",
        ]
        return "\n".join(lines)


def check(df: pl.DataFrame) -> QualityReport:
    """Inspect a bars frame and report anomalies. The frame is returned as-is
    by callers; nothing is dropped here."""
    report = QualityReport()
    for (symbol,), sym_df in df.sort("ts").group_by("symbol", maintain_order=True):
        sym = str(symbol)
        for row in sym_df.iter_rows(named=True):
            day = str(row["ts"].date())
            if row["volume"] == 0:
                report.zero_volume.append((sym, day))
            if row["close"] <= 0:
                report.nonpositive_close.append((sym, day))
        ts = sym_df["ts"].to_list()
        for prev, cur in itertools.pairwise(ts):
            # Calendar-day gap as a conservative proxy: > 7 calendar days is
            # more than 5 trading sessions in all but pathological weeks.
            gap_days = (cur - prev).days
            if gap_days > 7:
                report.gaps.append((sym, str(cur.date()), gap_days))
        closes = sym_df["close"].to_list()
        factors = sym_df["adj_factor"].to_list()
        for i in range(1, len(closes)):
            if closes[i - 1] <= 0:
                continue
            ret = closes[i] / closes[i - 1] - 1.0
            factor_changed = abs(factors[i] - factors[i - 1]) > 1e-12
            if abs(ret) > MAX_ABS_RETURN and not factor_changed:
                report.extreme_returns.append((sym, str(ts[i].date()), ret))
    if not report.clean:
        log.warning("Data quality issues found", summary=report.summary())
    return report
