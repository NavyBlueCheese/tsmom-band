"""The live runner.

Refuses to start if HALT exists. Refuses to touch a live port unless
I_UNDERSTAND_THIS_IS_REAL_MONEY is exactly "yes". One JSONL audit record per
decision. On any unhandled exception: cancel all open orders, write HALT,
exit non-zero.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import structlog
import typer

from twenty.backtest.types import PortfolioView
from twenty.data.sources import IBKRSource
from twenty.data.store import Catalog
from twenty.execution.broker import IBBroker
from twenty.execution.connection import IBSettings, connect
from twenty.execution.journal import Journal
from twenty.execution.reconcile import reconcile
from twenty.execution.schedule import next_trigger
from twenty.risk.checks import PreTradeRiskCheck
from twenty.risk.killswitch import MAX_CONSECUTIVE_REJECTIONS
from twenty.strategies.tsmom_band import TsmomBand, momentum, target_weights

log = structlog.get_logger(__name__)

app = typer.Typer(add_completion=False)

HALT_FILE = Path("HALT")
AUDIT_LOG = Path("logs") / "decisions.jsonl"


def audit(record: dict[str, Any]) -> None:
    AUDIT_LOG.parent.mkdir(exist_ok=True)
    record["logged_at"] = datetime.now(tz=UTC).isoformat()
    with AUDIT_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True, default=str) + "\n")


def assert_safe_to_start(settings: IBSettings) -> None:
    if HALT_FILE.exists():
        raise SystemExit(
            f"HALT file exists ({HALT_FILE.resolve()}). Delete it only after "
            "reading why it was written."
        )
    if not settings.is_paper:
        token = os.environ.get("I_UNDERSTAND_THIS_IS_REAL_MONEY", "")
        if token != "yes":
            raise SystemExit(
                "IB_IS_PAPER is false but I_UNDERSTAND_THIS_IS_REAL_MONEY is "
                "not exactly 'yes'. Refusing to start against real money."
            )


def rebalance_once(broker: IBBroker, catalog: Catalog, now: datetime) -> None:
    """One decision cycle: reconcile, decide, risk-check, transmit sells then
    buys, audit."""
    local_positions = broker.get_positions()
    broker_cash = broker.get_cash()
    reconcile(local_positions, broker_cash, broker.get_positions(), broker_cash)

    snapshot = catalog.as_of(now)
    view = PortfolioView(positions=local_positions, cash=broker_cash)
    strategy = TsmomBand()
    orders = strategy.on_bar(snapshot, view)
    checked = PreTradeRiskCheck().check(orders, view)

    record: dict[str, Any] = {
        "ts": now.isoformat(),
        "snapshot_hash": snapshot.hash(),
        "momentum": momentum(snapshot) if orders else None,
        "target_weights": (
            {k: str(v) for k, v in target_weights(snapshot).items()} if orders else None
        ),
        "positions": {k: str(v) for k, v in local_positions.items()},
        "cash": str(broker_cash),
        "orders_proposed": [o.model_dump(mode="json") for o in orders],
        "orders_after_risk": [o.model_dump(mode="json") for o in checked],
        "broker_responses": [],
        "fills": [],
    }
    fills: list[dict[str, Any]] = []
    broker.subscribe_fills(fills.append)
    for order in checked:  # sells already precede buys
        broker.place_order(order)
        if broker.consecutive_rejections >= MAX_CONSECUTIVE_REJECTIONS:
            HALT_FILE.write_text(
                f"{broker.consecutive_rejections} consecutive order rejections at "
                f"{datetime.now(tz=UTC).isoformat()}. Read the raw IBKR error "
                "codes in the log before anything else is sent.\n"
            )
            record["halted"] = "consecutive rejections"
            audit(record)
            raise SystemExit("Halted on consecutive order rejections")
    record["fills"] = fills
    audit(record)


# The daily Gateway logout/restart drops the socket while we are idle. That
# is expected, not fatal: keep retrying for hours (the box may need the
# Gateway's own auto-restart to finish, which takes minutes). Only give up —
# and HALT — if the gateway stays unreachable long enough that a human has
# clearly not noticed.
RECONNECT_INTERVAL_S = 60.0
RECONNECT_MAX_ATTEMPTS = 360  # 6 hours at one attempt per minute

# If the host slept through a trigger, executing late means trading outside
# the session on stale quotes. Beyond this lateness the trigger is skipped.
MAX_TRIGGER_LATENESS = timedelta(minutes=30)


def reconnect_with_patience(
    connect_fn: Callable[[], Any],
    max_attempts: int = RECONNECT_MAX_ATTEMPTS,
    interval_s: float = RECONNECT_INTERVAL_S,
    sleep_fn: Callable[[float], None] = time.sleep,
    halt_file: Path = HALT_FILE,
) -> Any:
    """Retry ``connect_fn`` until it succeeds or patience runs out. Safe only
    while idle: callers must not use this mid-order-flow."""
    for attempt in range(1, max_attempts + 1):
        if halt_file.exists():
            raise SystemExit(f"HALT file appeared during reconnect; stopping. {halt_file}")
        try:
            ib = connect_fn()
        except (ConnectionError, OSError, TimeoutError) as exc:
            log.warning(
                "Gateway unreachable, will retry",
                attempt=attempt,
                max_attempts=max_attempts,
                error=str(exc),
            )
            sleep_fn(interval_s)
            continue
        log.info("Reconnected to gateway", attempt=attempt)
        return ib
    halt_file.write_text(
        f"Gateway unreachable for {max_attempts * interval_s / 3600:.1f} hours "
        f"as of {datetime.now(tz=UTC).isoformat()}. Log into IB Gateway, then "
        "delete this file.\n"
    )
    raise SystemExit("Gateway unreachable past patience; HALT written")


@app.command()
def main(
    force_weekly: bool = typer.Option(
        False, help="Paper testing only: trigger every Friday instead of quarter ends"
    ),
) -> None:
    settings = IBSettings()
    assert_safe_to_start(settings)
    if force_weekly and not settings.is_paper:
        raise SystemExit("--force-weekly is a paper-testing flag; refusing on live")

    ib = connect(settings)
    journal = Journal(Path("journal.sqlite"))
    account = settings.account or next(iter(ib.managedAccounts()), "")
    if not account:
        raise SystemExit("No account reported by the gateway; cannot start")
    log.info("Trading account resolved", account=account)
    broker = IBBroker(ib, journal, account)

    # Startup reconcile before anything else: journal vs openOrders/trades.
    open_ids = {t.order.orderId for t in ib.openTrades()}
    filled_ids = {
        t.order.orderId for t in ib.trades() if t.orderStatus.status == "Filled"
    }
    live = journal.reconcile_on_startup(open_ids, filled_ids)
    if live:
        log.warning("Orders still live after startup reconcile", count=len(live))

    try:
        while True:
            now = datetime.now(tz=UTC)
            trigger = next_trigger(now, force_weekly=force_weekly)
            log.info("Sleeping until trigger", trigger=trigger.isoformat())
            while datetime.now(tz=UTC) < trigger:
                if HALT_FILE.exists():
                    raise SystemExit(f"HALT file appeared; stopping. {HALT_FILE.resolve()}")
                try:
                    if not ib.isConnected():
                        raise ConnectionError("gateway disconnected")
                    ib.sleep(30)
                except (ConnectionError, OSError, TimeoutError) as exc:
                    # The daily Gateway logout is expected, not fatal. We are
                    # idle (orders only exist inside rebalance_once), so:
                    # reconnect patiently, reconcile, only then resume.
                    log.warning("Connection lost while idle", error=str(exc))
                    ib = reconnect_with_patience(lambda: connect(settings))
                    broker = IBBroker(ib, journal, account)
                    reconcile(
                        broker.get_positions(),
                        broker.get_cash(),
                        broker.get_positions(),
                        broker.get_cash(),
                    )
            lateness = datetime.now(tz=UTC) - trigger
            if lateness > MAX_TRIGGER_LATENESS:
                # The host slept through the trigger. Trading hours late on
                # stale prices is worse than not trading; a quarterly system
                # can wait for its next scheduled session.
                log.warning(
                    "Missed trigger by too much, skipping to the next one",
                    trigger=trigger.isoformat(),
                    late_by_minutes=round(lateness.total_seconds() / 60),
                )
                continue
            source = IBKRSource(settings.host, settings.port)
            end = datetime.now(tz=UTC).date()
            bars = source.fetch(
                ["SPY", "EFA", "IEF", "GLD"],
                end.replace(year=end.year - 2),
                end,
            )
            catalog = Catalog(bars)
            rebalance_once(broker, catalog, datetime.now(tz=UTC))
    except SystemExit:
        raise
    except BaseException as exc:
        log.error("Unhandled exception; cancelling orders and writing HALT", error=str(exc))
        try:
            broker.cancel_all_open()
        finally:
            HALT_FILE.write_text(
                f"Unhandled exception at {datetime.now(tz=UTC).isoformat()}: "
                f"{exc!r}\n"
            )
        sys.exit(1)


if __name__ == "__main__":
    app()
