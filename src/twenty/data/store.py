"""Parquet store, catalog, and the point-in-time Snapshot.

Snapshot is the single most important class in the repo: it is the only thing
a strategy ever sees, and it is constructed so that data after its timestamp
is physically absent, not merely hidden.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType

import numpy as np
import polars as pl

from twenty.data.schema import BAR_COLUMNS


class ParquetStore:
    """Bars on disk, partitioned symbol/year."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def write(self, df: pl.DataFrame) -> None:
        for (symbol,), sym_df in df.group_by("symbol"):
            for (year,), year_df in sym_df.group_by(pl.col("ts").dt.year()):
                out_dir = self.root / str(symbol) / str(year)
                out_dir.mkdir(parents=True, exist_ok=True)
                year_df.sort("ts").write_parquet(out_dir / "bars.parquet")

    def read(self) -> pl.DataFrame:
        files = sorted(self.root.rglob("*.parquet"))
        if not files:
            return pl.DataFrame({c: [] for c in BAR_COLUMNS})
        return pl.concat([pl.read_parquet(f) for f in files]).sort(["symbol", "ts"])


@dataclass(frozen=True)
class SymbolHistory:
    """Bounded, already-sliced arrays for one symbol, ending at the snapshot ts.

    Arrays are copies with the writeable flag cleared; they hold no reference
    to the catalog's full arrays.
    """

    ts_ns: np.ndarray  # int64 epoch nanoseconds, ascending
    open: np.ndarray
    close: np.ndarray
    adj_factor: np.ndarray

    def __len__(self) -> int:
        return int(self.ts_ns.shape[0])


@dataclass(frozen=True)
class Snapshot:
    """Read-only view of all history up to and including ``ts``.

    Holds only bounded copies. It must not hold a reference to the full
    frame, a full index, or a closure over either — by construction the
    only arrays present were sliced and copied before this object existed.
    """

    ts: datetime
    _data: Mapping[str, SymbolHistory] = field(repr=False)

    def symbols(self) -> tuple[str, ...]:
        return tuple(sorted(self._data))

    def sessions(self, symbol: str) -> int:
        return len(self._data[symbol])

    def raw_close(self, symbol: str) -> np.ndarray:
        return self._data[symbol].close

    def adjusted_close(self, symbol: str) -> np.ndarray:
        """Total-return-adjusted closes, rebased so the last visible bar's
        adjusted price equals its raw price.

        adj[i] = close[i] * adj_factor[i] / adj_factor[-1]. Because
        adj_factor[i] is the cumulative action factor as of bar i, the ratio
        uses only actions that occurred at or before ``ts``.
        """
        h = self._data[symbol]
        if len(h) == 0:
            return np.empty(0, dtype=np.float64)
        out: np.ndarray = h.close * (h.adj_factor / h.adj_factor[-1])
        out.flags.writeable = False
        return out

    def hash(self) -> str:
        """Deterministic digest of everything visible in this snapshot."""
        digest = hashlib.sha256()
        digest.update(str(self.ts).encode())
        for symbol in self.symbols():
            h = self._data[symbol]
            digest.update(symbol.encode())
            digest.update(h.ts_ns.tobytes())
            digest.update(h.close.tobytes())
            digest.update(h.adj_factor.tobytes())
        return digest.hexdigest()


def _ro(arr: np.ndarray) -> np.ndarray:
    out = np.ascontiguousarray(arr)
    out.flags.writeable = False
    return out


class Catalog:
    """All loaded history, with point-in-time slicing.

    The engine may consult the catalog for t+1 fills and corporate actions;
    the strategy only ever receives Snapshots.
    """

    def __init__(self, df: pl.DataFrame) -> None:
        if df.is_empty():
            raise ValueError("Catalog requires at least one bar")
        df = df.sort(["symbol", "ts"])
        self._arrays: dict[str, dict[str, np.ndarray]] = {}
        for (symbol,), sym_df in df.group_by("symbol", maintain_order=True):
            ts_ns = (
                sym_df["ts"].dt.convert_time_zone("UTC").dt.timestamp("ns").to_numpy()
            )
            self._arrays[str(symbol)] = {
                "ts_ns": _ro(ts_ns.astype(np.int64)),
                "open": _ro(sym_df["open"].to_numpy().astype(np.float64)),
                "close": _ro(sym_df["close"].to_numpy().astype(np.float64)),
                "adj_factor": _ro(sym_df["adj_factor"].to_numpy().astype(np.float64)),
                "dividend": _ro(sym_df["dividend"].to_numpy().astype(np.float64)),
                "split": _ro(sym_df["split"].to_numpy().astype(np.float64)),
            }
        all_ts = np.unique(
            np.concatenate([a["ts_ns"] for a in self._arrays.values()])
        )
        self._sessions_ns: np.ndarray = _ro(all_ts)

    @classmethod
    def from_dir(cls, root: Path) -> Catalog:
        return cls(ParquetStore(root).read())

    @staticmethod
    def _to_ns(ts: datetime) -> int:
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        return int(ts.timestamp() * 1_000_000_000)

    @staticmethod
    def _from_ns(ns: int) -> datetime:
        return datetime.fromtimestamp(ns / 1_000_000_000, tz=UTC)

    def sessions(self) -> list[datetime]:
        return [self._from_ns(int(ns)) for ns in self._sessions_ns]

    def next_session(self, ts: datetime) -> datetime | None:
        ns = self._to_ns(ts)
        idx = int(np.searchsorted(self._sessions_ns, ns, side="right"))
        if idx >= self._sessions_ns.shape[0]:
            return None
        return self._from_ns(int(self._sessions_ns[idx]))

    def as_of(self, ts: datetime) -> Snapshot:
        """Snapshot of everything at or before ``ts``. Copies, never views."""
        ns = self._to_ns(ts)
        data: dict[str, SymbolHistory] = {}
        for symbol, arrays in self._arrays.items():
            n = int(np.searchsorted(arrays["ts_ns"], ns, side="right"))
            data[symbol] = SymbolHistory(
                ts_ns=_ro(arrays["ts_ns"][:n].copy()),
                open=_ro(arrays["open"][:n].copy()),
                close=_ro(arrays["close"][:n].copy()),
                adj_factor=_ro(arrays["adj_factor"][:n].copy()),
            )
        return Snapshot(ts=ts, _data=MappingProxyType(data))

    def open_at(self, symbol: str, ts: datetime) -> float | None:
        arrays = self._arrays[symbol]
        ns = self._to_ns(ts)
        idx = int(np.searchsorted(arrays["ts_ns"], ns, side="left"))
        if idx >= arrays["ts_ns"].shape[0] or int(arrays["ts_ns"][idx]) != ns:
            return None
        return float(arrays["open"][idx])

    def close_at(self, symbol: str, ts: datetime) -> float | None:
        arrays = self._arrays[symbol]
        ns = self._to_ns(ts)
        idx = int(np.searchsorted(arrays["ts_ns"], ns, side="left"))
        if idx >= arrays["ts_ns"].shape[0] or int(arrays["ts_ns"][idx]) != ns:
            return None
        return float(arrays["close"][idx])

    def actions_at(self, symbol: str, ts: datetime) -> tuple[float, float]:
        """(dividend per share, split multiplier) going effective on ``ts``."""
        arrays = self._arrays[symbol]
        ns = self._to_ns(ts)
        idx = int(np.searchsorted(arrays["ts_ns"], ns, side="left"))
        if idx >= arrays["ts_ns"].shape[0] or int(arrays["ts_ns"][idx]) != ns:
            return (0.0, 1.0)
        return (float(arrays["dividend"][idx]), float(arrays["split"][idx]))
