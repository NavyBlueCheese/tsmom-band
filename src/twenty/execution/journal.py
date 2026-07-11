"""Order journal: SQLite in WAL mode, written BEFORE transmission.

This exists because some IBKR errors (10349 among them) surface as errors to
the client while leaving the order live at the broker. If you treat those as
failures and retry, you double your position. The journal makes every order
idempotent: an order whose UUID already exists in a non-terminal state is
never transmitted again.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from uuid import UUID

import structlog

from twenty.backtest.types import ProposedOrder

log = structlog.get_logger(__name__)


class OrderState(StrEnum):
    PENDING = "PENDING"
    TRANSMITTED = "TRANSMITTED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


TERMINAL_STATES = frozenset({OrderState.FILLED, OrderState.CANCELLED, OrderState.REJECTED})

_SCHEMA = """
CREATE TABLE IF NOT EXISTS orders (
    client_id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    notional TEXT NOT NULL,
    order_type TEXT NOT NULL,
    state TEXT NOT NULL,
    broker_order_id INTEGER,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


class Journal:
    def __init__(self, path: Path) -> None:
        self._conn = sqlite3.connect(path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def record_proposed(self, order: ProposedOrder) -> None:
        """Write the order as PENDING. Must be called (and committed) before
        any transmission attempt."""
        now = datetime.now(tz=UTC).isoformat()
        self._conn.execute(
            "INSERT INTO orders (client_id, symbol, side, notional, order_type, "
            "state, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(order.client_id),
                order.symbol,
                order.side.value,
                str(order.notional),
                order.order_type.value,
                OrderState.PENDING.value,
                now,
                now,
            ),
        )
        self._conn.commit()

    def set_state(
        self, client_id: UUID, state: OrderState, broker_order_id: int | None = None
    ) -> None:
        now = datetime.now(tz=UTC).isoformat()
        if broker_order_id is not None:
            self._conn.execute(
                "UPDATE orders SET state = ?, broker_order_id = ?, updated_at = ? "
                "WHERE client_id = ?",
                (state.value, broker_order_id, now, str(client_id)),
            )
        else:
            self._conn.execute(
                "UPDATE orders SET state = ?, updated_at = ? WHERE client_id = ?",
                (state.value, now, str(client_id)),
            )
        self._conn.commit()

    def state_of(self, client_id: UUID) -> OrderState | None:
        row = self._conn.execute(
            "SELECT state FROM orders WHERE client_id = ?", (str(client_id),)
        ).fetchone()
        return OrderState(row[0]) if row else None

    def non_terminal(self) -> list[tuple[UUID, str, OrderState]]:
        rows = self._conn.execute(
            "SELECT client_id, symbol, state FROM orders WHERE state NOT IN (?, ?, ?)",
            tuple(s.value for s in TERMINAL_STATES),
        ).fetchall()
        return [(UUID(r[0]), r[1], OrderState(r[2])) for r in rows]

    def may_transmit(self, order: ProposedOrder) -> bool:
        """False if this UUID already exists in any non-terminal state — the
        anti-double-position property."""
        state = self.state_of(order.client_id)
        if state is None:
            return True
        if state in TERMINAL_STATES:
            log.warning(
                "Order UUID already terminal, not retransmitting",
                client_id=str(order.client_id),
                state=state.value,
            )
            return False
        log.warning(
            "Order UUID already live at broker, not retransmitting",
            client_id=str(order.client_id),
            state=state.value,
        )
        return False

    def reconcile_on_startup(
        self, open_broker_ids: set[int], filled_broker_ids: set[int]
    ) -> list[UUID] :
        """Cross-check journal against ib.openOrders()/ib.trades() after a
        restart. Orders journalled TRANSMITTED that the broker reports filled
        become FILLED; ones the broker no longer knows become CANCELLED.
        Returns UUIDs still live at the broker."""
        still_live: list[UUID] = []
        rows = self._conn.execute(
            "SELECT client_id, broker_order_id, state FROM orders "
            "WHERE state IN (?, ?)",
            (OrderState.PENDING.value, OrderState.TRANSMITTED.value),
        ).fetchall()
        for client_id_s, broker_order_id, state in rows:
            client_id = UUID(client_id_s)
            if broker_order_id is not None and broker_order_id in open_broker_ids:
                still_live.append(client_id)
            elif broker_order_id is not None and broker_order_id in filled_broker_ids:
                self.set_state(client_id, OrderState.FILLED)
            elif state == OrderState.PENDING.value and broker_order_id is None:
                # Journalled but never transmitted: safe to cancel locally.
                self.set_state(client_id, OrderState.CANCELLED)
            else:
                # Transmitted, broker has no record open or filled. Do NOT
                # assume dead: leave it non-terminal so may_transmit refuses a
                # duplicate, and let a human look.
                still_live.append(client_id)
                log.warning(
                    "Transmitted order unknown to broker after restart; "
                    "leaving non-terminal pending human review",
                    client_id=client_id_s,
                )
        return still_live
