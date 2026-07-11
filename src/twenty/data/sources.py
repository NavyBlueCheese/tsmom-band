"""Data sources behind a common protocol."""

from __future__ import annotations

import threading
import time
from collections import deque
from datetime import UTC, date, datetime
from typing import Any, Protocol

import polars as pl
import structlog

from twenty.data.schema import BAR_COLUMNS

log = structlog.get_logger(__name__)


class DataSource(Protocol):
    """Anything that can fetch daily bars for a list of symbols."""

    def fetch(self, symbols: list[str], start: date, end: date) -> pl.DataFrame:
        """Return a frame with BAR_COLUMNS, one row per symbol per session."""
        ...


class YFinanceSource:
    """Yahoo Finance daily bars, for research bootstrap only.

    Survivorship note: Yahoo only serves symbols that still exist, so any
    process that *selects a universe from history* using this source is
    survivorship-biased. Our universe is four fixed, still-listed ETFs chosen
    in the preregistration, not selected from historical data, so
    survivorship bias does not apply to this project. The constructor warning
    stays anyway: it is aimed at the next project that copies this file.
    """

    def __init__(self) -> None:
        log.warning(
            "YFinance source is research only: survivorship-biased, unsuitable "
            "for anything that selects a universe from history"
        )

    def fetch(self, symbols: list[str], start: date, end: date) -> pl.DataFrame:
        import yfinance as yf

        frames: list[pl.DataFrame] = []
        for symbol in symbols:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(
                start=start.isoformat(),
                end=end.isoformat(),
                auto_adjust=False,
                actions=True,
            )
            if hist.empty:
                log.warning("No data returned", symbol=symbol)
                continue
            hist = hist.reset_index()
            rows: dict[str, list[Any]] = {c: [] for c in BAR_COLUMNS}
            # Cumulative adjustment factor as of each date, reconstructed
            # point-in-time from the dividend and split event streams so it
            # embeds no knowledge of actions after the bar's own date.
            factor = 1.0
            for rec in hist.to_dict("records"):
                ts = rec["Date"].to_pydatetime()
                ts = ts.replace(tzinfo=UTC) if ts.tzinfo is None else ts.astimezone(UTC)
                close = float(rec["Close"])
                dividend = float(rec.get("Dividends", 0.0) or 0.0)
                split = float(rec.get("Stock Splits", 0.0) or 0.0)
                if split == 0.0:
                    split = 1.0
                if split != 1.0:
                    factor *= split
                if dividend > 0.0 and close > 0.0:
                    prev_close = close + dividend  # approximate prior close ex the dividend
                    factor *= prev_close / close if close > 0 else 1.0
                rows["symbol"].append(symbol)
                rows["ts"].append(ts)
                rows["open"].append(float(rec["Open"]))
                rows["high"].append(float(rec["High"]))
                rows["low"].append(float(rec["Low"]))
                rows["close"].append(close)
                rows["volume"].append(float(rec["Volume"]))
                rows["adj_factor"].append(factor)
                rows["dividend"].append(dividend)
                rows["split"].append(split)
            frames.append(pl.DataFrame(rows))
        if not frames:
            return pl.DataFrame({c: [] for c in BAR_COLUMNS})
        return pl.concat(frames).sort(["symbol", "ts"])


class TokenBucket:
    """Rate limiter: at most ``capacity`` requests per ``window_s`` seconds,
    plus a minimum spacing between identical request keys."""

    def __init__(
        self,
        capacity: int = 60,
        window_s: float = 600.0,
        min_identical_gap_s: float = 2.0,
        clock: Any = time,
    ) -> None:
        self.capacity = capacity
        self.window_s = window_s
        self.min_identical_gap_s = min_identical_gap_s
        self._clock = clock
        self._stamps: deque[float] = deque()
        self._last_by_key: dict[str, float] = {}
        self._lock = threading.Lock()

    def acquire(self, key: str) -> float:
        """Block until a request with ``key`` may proceed. Returns seconds slept."""
        slept = 0.0
        with self._lock:
            while True:
                now = self._clock.time()
                while self._stamps and now - self._stamps[0] > self.window_s:
                    self._stamps.popleft()
                wait = 0.0
                if len(self._stamps) >= self.capacity:
                    wait = max(wait, self.window_s - (now - self._stamps[0]))
                last = self._last_by_key.get(key)
                if last is not None:
                    gap = now - last
                    if gap < self.min_identical_gap_s:
                        wait = max(wait, self.min_identical_gap_s - gap)
                if wait <= 0.0:
                    self._stamps.append(now)
                    self._last_by_key[key] = now
                    return slept
                self._clock.sleep(wait)
                slept += wait


class IBKRSource:
    """Historical daily bars from IB Gateway via ib_async reqHistoricalData."""

    def __init__(self, host: str = "127.0.0.1", port: int = 4002, client_id: int = 21) -> None:
        self._host = host
        self._port = port
        self._client_id = client_id
        self._bucket = TokenBucket(capacity=60, window_s=600.0, min_identical_gap_s=2.0)

    def fetch(self, symbols: list[str], start: date, end: date) -> pl.DataFrame:
        from ib_async import IB, Stock, util

        ib = IB()
        ib.connect(self._host, self._port, clientId=self._client_id, readonly=True)
        # Delayed data: without a live market data subscription (typical on
        # paper accounts) historical requests are otherwise refused.
        ib.reqMarketDataType(3)
        try:
            frames: list[pl.DataFrame] = []
            duration_days = (end - start).days + 1
            duration = f"{max(duration_days, 1)} D"
            end_dt = datetime.combine(end, datetime.min.time(), tzinfo=UTC)
            for symbol in symbols:
                key = f"{symbol}:{start}:{end}"
                self._bucket.acquire(key)
                contract = Stock(symbol, "SMART", "USD")
                bars = ib.reqHistoricalData(
                    contract,
                    endDateTime=end_dt,
                    durationStr=duration,
                    barSizeSetting="1 day",
                    whatToShow="TRADES",
                    useRTH=True,
                    formatDate=2,
                )
                if not bars:
                    log.warning("No IBKR data returned", symbol=symbol)
                    continue
                df = util.df(bars)
                frames.append(
                    pl.DataFrame(
                        {
                            "symbol": [symbol] * len(df),
                            "ts": [
                                datetime.combine(d, datetime.min.time(), tzinfo=UTC)
                                for d in df["date"]
                            ],
                            "open": df["open"].astype(float).tolist(),
                            "high": df["high"].astype(float).tolist(),
                            "low": df["low"].astype(float).tolist(),
                            "close": df["close"].astype(float).tolist(),
                            "volume": df["volume"].astype(float).tolist(),
                            # IBKR TRADES bars are unadjusted; corporate action
                            # streams must be merged separately if this source
                            # is ever used for research history.
                            "adj_factor": [1.0] * len(df),
                            "dividend": [0.0] * len(df),
                            "split": [1.0] * len(df),
                        }
                    )
                )
            if not frames:
                return pl.DataFrame({c: [] for c in BAR_COLUMNS})
            return pl.concat(frames).sort(["symbol", "ts"])
        finally:
            ib.disconnect()
