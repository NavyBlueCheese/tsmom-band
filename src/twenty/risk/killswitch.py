"""The kill switch.

Checks run in a fixed order because the earlier ones must work when the later
subsystems don't: a HALT file requires nothing but a filesystem; reconciliation
requires broker connectivity; staleness requires market data; loss and
drawdown require a working ledger; rejection streaks require order flow.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from pathlib import Path


class HaltCause(StrEnum):
    HALT_FILE = "HALT_FILE"
    RECONCILIATION_MISMATCH = "RECONCILIATION_MISMATCH"
    STALE_MARKET_DATA = "STALE_MARKET_DATA"
    DAILY_LOSS = "DAILY_LOSS"
    DRAWDOWN = "DRAWDOWN"
    CONSECUTIVE_REJECTIONS = "CONSECUTIVE_REJECTIONS"


@dataclass(frozen=True)
class HaltReason:
    cause: HaltCause
    detail: str

    def __str__(self) -> str:
        return f"HALT [{self.cause.value}]: {self.detail}"


@dataclass(frozen=True)
class PortfolioState:
    equity: Decimal
    day_start_equity: Decimal
    high_water_mark: Decimal
    positions: dict[str, Decimal]
    cash: Decimal


@dataclass(frozen=True)
class BrokerState:
    positions: dict[str, Decimal]
    cash: Decimal
    last_tick_at: datetime | None
    is_rth: bool
    consecutive_rejections: int


RECONCILIATION_TOLERANCE = Decimal("0.02")
STALENESS_LIMIT_S = 300.0
MAX_CONSECUTIVE_REJECTIONS = 3


def should_halt(
    portfolio: PortfolioState,
    broker: BrokerState,
    max_daily_loss_pct: float = 0.10,
    max_drawdown_pct: float = 0.35,
    halt_file: Path = Path("HALT"),
    now: datetime | None = None,
) -> HaltReason | None:
    now = now or datetime.now(tz=UTC)

    # 1. HALT file. Works when nothing else does.
    if halt_file.exists():
        content = halt_file.read_text(errors="replace").strip() or "(empty)"
        return HaltReason(
            HaltCause.HALT_FILE,
            f"File {halt_file.resolve()} exists with content: {content}. "
            "A human asked for a stop; nothing trades until it is deleted.",
        )

    # 2. Reconciliation against the broker.
    cash_diff = abs(portfolio.cash - broker.cash)
    if cash_diff > RECONCILIATION_TOLERANCE:
        return HaltReason(
            HaltCause.RECONCILIATION_MISMATCH,
            f"Cash mismatch of ${cash_diff}: local ${portfolio.cash} vs "
            f"broker ${broker.cash}. Books disagree with the broker; do not "
            "trade until a human reconciles the journal against ib.trades().",
        )
    for symbol in sorted(set(portfolio.positions) | set(broker.positions)):
        local = portfolio.positions.get(symbol, Decimal(0))
        remote = broker.positions.get(symbol, Decimal(0))
        # Position mismatch measured in shares; $0.02 of any ETF here is a
        # fraction of a share, so any share difference beyond rounding halts.
        if abs(local - remote) > Decimal("0.0002"):
            return HaltReason(
                HaltCause.RECONCILIATION_MISMATCH,
                f"Position mismatch in {symbol}: local {local} shares vs "
                f"broker {remote} shares. Books disagree with the broker; do "
                "not trade until a human reconciles the journal.",
            )

    # 3. Market data staleness during regular trading hours.
    if broker.is_rth:
        if broker.last_tick_at is None:
            return HaltReason(
                HaltCause.STALE_MARKET_DATA,
                "No market data ticks received at all during regular trading "
                "hours. Quotes cannot be trusted; refusing to price orders.",
            )
        age = (now - broker.last_tick_at).total_seconds()
        if age > STALENESS_LIMIT_S:
            return HaltReason(
                HaltCause.STALE_MARKET_DATA,
                f"Newest tick is {age:.0f}s old (limit {STALENESS_LIMIT_S:.0f}s) "
                "during regular trading hours. Quotes cannot be trusted.",
            )

    # 4. Daily loss.
    if portfolio.day_start_equity > 0:
        daily_loss = float(
            (portfolio.day_start_equity - portfolio.equity) / portfolio.day_start_equity
        )
        if daily_loss > max_daily_loss_pct:
            return HaltReason(
                HaltCause.DAILY_LOSS,
                f"Down {daily_loss:.1%} today (limit {max_daily_loss_pct:.0%}): "
                f"equity ${portfolio.equity} from ${portfolio.day_start_equity} "
                "at the open. Something is wrong with sizing or the market; "
                "stop and look.",
            )

    # 5. Drawdown from high-water mark.
    if portfolio.high_water_mark > 0:
        drawdown = float(
            (portfolio.high_water_mark - portfolio.equity) / portfolio.high_water_mark
        )
        if drawdown > max_drawdown_pct:
            return HaltReason(
                HaltCause.DRAWDOWN,
                f"Drawdown {drawdown:.1%} from high-water mark "
                f"${portfolio.high_water_mark} (limit {max_drawdown_pct:.0%}), "
                f"equity ${portfolio.equity}. The preregistered risk budget is "
                "spent; the strategy does not get to keep digging.",
            )

    # 6. Rejection streak.
    if broker.consecutive_rejections >= MAX_CONSECUTIVE_REJECTIONS:
        return HaltReason(
            HaltCause.CONSECUTIVE_REJECTIONS,
            f"{broker.consecutive_rejections} consecutive order rejections "
            f"(limit {MAX_CONSECUTIVE_REJECTIONS}). The broker keeps saying no "
            "and retrying is how positions double; a human reads the rejection "
            "codes before anything else is sent.",
        )

    return None
