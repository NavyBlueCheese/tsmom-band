"""Reconciliation: local books versus the broker's. Never auto-correct."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import structlog

log = structlog.get_logger(__name__)

TOLERANCE = Decimal("0.02")


class ReconciliationError(RuntimeError):
    pass


def reconcile(
    local_positions: dict[str, Decimal],
    local_cash: Decimal,
    broker_positions: dict[str, Decimal],
    broker_cash: Decimal,
    halt_file: Path = Path("HALT"),
) -> None:
    """Compare local state to the broker's. Any mismatch beyond $0.02 writes
    HALT and raises. Nothing is ever auto-corrected: the fix is a human
    reading the journal and ib.trades() and editing in Client Portal."""
    problems: list[str] = []
    if abs(local_cash - broker_cash) > TOLERANCE:
        problems.append(
            f"cash: local ${local_cash} vs broker ${broker_cash} "
            f"(diff ${abs(local_cash - broker_cash)})"
        )
    for symbol in sorted(set(local_positions) | set(broker_positions)):
        local = local_positions.get(symbol, Decimal(0))
        remote = broker_positions.get(symbol, Decimal(0))
        if abs(local - remote) > Decimal("0.0002"):
            problems.append(f"{symbol}: local {local} shares vs broker {remote} shares")
    if problems:
        detail = "; ".join(problems)
        halt_file.write_text(f"Reconciliation mismatch: {detail}\n")
        log.error("Reconciliation failed, HALT written", detail=detail)
        raise ReconciliationError(detail)
    log.info("Reconciliation clean")
