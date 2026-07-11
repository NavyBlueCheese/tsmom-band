"""Bootstrap CLI: download history, split train/holdout, seal the holdout.

    python -m twenty.data.bootstrap --start 2005-01-01

GLD's inception is late 2004; earlier data does not exist and is never
backfilled with anything.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import structlog
import typer

from twenty.data import quality
from twenty.data.sources import YFinanceSource
from twenty.data.store import ParquetStore

log = structlog.get_logger(__name__)

UNIVERSE = ["SPY", "EFA", "IEF", "GLD"]
HOLDOUT_START = date(2019, 1, 1)

app = typer.Typer(add_completion=False)


def holdout_digest(holdout_dir: Path) -> str:
    """SHA-256 over every parquet file in the holdout directory, in sorted
    path order, so any change to sealed data changes the digest."""
    digest = hashlib.sha256()
    for f in sorted(holdout_dir.rglob("*.parquet")):
        digest.update(str(f.relative_to(holdout_dir)).encode())
        digest.update(f.read_bytes())
    return digest.hexdigest()


@app.command()
def main(
    start: str = typer.Option("2005-01-01", help="First date to download"),
    root: Path = typer.Option(Path("data"), help="Data root directory"),
) -> None:
    start_date = date.fromisoformat(start)
    end_date = datetime.now(tz=UTC).date()
    source = YFinanceSource()
    log.info("Downloading", symbols=UNIVERSE, start=str(start_date), end=str(end_date))
    df = source.fetch(UNIVERSE, start_date, end_date)
    if df.is_empty():
        log.error("Download returned no data")
        raise typer.Exit(code=1)

    report = quality.check(df)
    print("Data quality report:")
    print(report.summary())

    ParquetStore(root / "raw").write(df)
    cutoff = datetime(
        HOLDOUT_START.year, HOLDOUT_START.month, HOLDOUT_START.day, tzinfo=UTC
    )
    train = df.filter(pl.col("ts") < cutoff)
    holdout = df.filter(pl.col("ts") >= cutoff)
    ParquetStore(root / "train").write(train)
    ParquetStore(root / "holdout").write(holdout)

    digest = holdout_digest(root / "holdout")
    (root / "holdout" / "HASH").write_text(digest + "\n")
    print(f"Rows: raw={len(df)} train={len(train)} holdout={len(holdout)}")
    print(f"Holdout SHA-256: {digest}")


if __name__ == "__main__":
    app()
