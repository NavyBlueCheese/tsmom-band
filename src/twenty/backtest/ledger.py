"""Append-only ledger: one row per session, replayable via the snapshot hash."""

from __future__ import annotations

import json
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import polars as pl

from twenty.backtest.types import Fill, ProposedOrder


def _order_record(order: ProposedOrder) -> dict[str, str | int]:
    return {
        "symbol": order.symbol,
        "side": order.side.value,
        "notional": str(order.notional),
        "order_type": order.order_type.value,
        "limit_offset_bps": order.limit_offset_bps,
        "client_id": str(order.client_id),
    }


def _fill_record(fill: Fill) -> dict[str, str]:
    return {
        "symbol": fill.symbol,
        "side": fill.side.value,
        "shares": str(fill.shares),
        "price": str(fill.price),
        "ts": fill.ts.isoformat(),
        "commission": str(fill.commission),
        "slippage": str(fill.slippage),
        "client_id": str(fill.client_id),
    }


class Ledger:
    """In-memory append-only rows with canonical JSON serialisation, written
    to parquet at the end of a run."""

    def __init__(self) -> None:
        self._rows: list[dict[str, Any]] = []

    def record(
        self,
        ts: datetime,
        snapshot_hash: str,
        orders: list[ProposedOrder],
        fills: list[Fill],
        cash: Decimal,
        positions: dict[str, Decimal],
        equity: Decimal,
    ) -> None:
        self._rows.append(
            {
                "ts": ts.isoformat(),
                "snapshot_hash": snapshot_hash,
                "orders": json.dumps([_order_record(o) for o in orders], sort_keys=True),
                "fills": json.dumps([_fill_record(f) for f in fills], sort_keys=True),
                "cash": str(cash),
                "positions": json.dumps(
                    {s: str(q) for s, q in sorted(positions.items()) if q != 0},
                    sort_keys=True,
                ),
                "equity": str(equity),
            }
        )

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def rows(self) -> list[dict[str, Any]]:
        return list(self._rows)

    def canonical_row(self, i: int) -> bytes:
        """Deterministic byte serialisation of one row, minus the random
        client UUIDs (regenerated per run, semantically irrelevant)."""
        row = dict(self._rows[i])
        for key in ("orders", "fills"):
            items = json.loads(row[key])
            for item in items:
                item.pop("client_id", None)
            row[key] = json.dumps(items, sort_keys=True)
        return json.dumps(row, sort_keys=True).encode()

    def fills(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for row in self._rows:
            out.extend(json.loads(row["fills"]))
        return out

    def orders(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for row in self._rows:
            out.extend(json.loads(row["orders"]))
        return out

    def equity_series(self) -> list[tuple[datetime, float]]:
        return [
            (datetime.fromisoformat(r["ts"]), float(r["equity"])) for r in self._rows
        ]

    def total_commission(self) -> Decimal:
        total = Decimal(0)
        for fill in self.fills():
            total += Decimal(fill["commission"])
        return total

    def total_slippage(self) -> Decimal:
        total = Decimal(0)
        for fill in self.fills():
            total += Decimal(fill["slippage"])
        return total

    def write_parquet(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(self._rows).write_parquet(path)
