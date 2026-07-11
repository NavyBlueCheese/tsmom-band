"""Trial logging: the honesty mechanism.

Every configuration ever backtested is appended to trials.jsonl with a hash
of its config. The tearsheet refuses to build if the trial count is lower
than the number of distinct configs found in the ledger directory, and the
deflated Sharpe reads its n_trials from here. Do not make this bypassable.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ParamSpec

TRIALS_PATH = Path("trials.jsonl")

P = ParamSpec("P")


def config_hash(config: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


def log_trial(
    config: dict[str, Any], metrics: dict[str, Any], path: Path = TRIALS_PATH
) -> None:
    record = {
        "ts": datetime.now(tz=UTC).isoformat(),
        "config_hash": config_hash(config),
        "config": {k: str(v) for k, v in config.items()},
        "metrics": {k: str(v) for k, v in metrics.items()},
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def trial(
    path: Path = TRIALS_PATH,
) -> Callable[[Callable[P, dict[str, Any]]], Callable[P, dict[str, Any]]]:
    """Decorator: the wrapped function receives a ``config`` kwarg and returns
    a metrics dict; every invocation is appended to trials.jsonl."""

    def decorate(func: Callable[P, dict[str, Any]]) -> Callable[P, dict[str, Any]]:
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> dict[str, Any]:
            config = kwargs.get("config")
            if not isinstance(config, dict):
                raise TypeError("@trial functions must be called with a config= dict")
            metrics = func(*args, **kwargs)
            log_trial(config, metrics, path=path)
            return metrics

        return wrapper

    return decorate


def trial_count(path: Path = TRIALS_PATH) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def distinct_config_count(path: Path = TRIALS_PATH) -> int:
    if not path.exists():
        return 0
    hashes = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                hashes.add(json.loads(line)["config_hash"])
    return len(hashes)


def ledger_config_count(ledger_dir: Path) -> int:
    """Distinct configurations that produced ledgers on disk."""
    if not ledger_dir.exists():
        return 0
    return len(list(ledger_dir.glob("*.parquet")))


def assert_trials_cover_ledgers(
    ledger_dir: Path, trials_path: Path = TRIALS_PATH
) -> int:
    """The honesty check. Returns the trial count, or raises."""
    if not trials_path.exists():
        raise RuntimeError(
            f"{trials_path} is absent. Every backtested configuration must be "
            "logged; refusing to build anything that reports results."
        )
    n_trials = trial_count(trials_path)
    n_ledgers = ledger_config_count(ledger_dir)
    if n_trials < n_ledgers:
        raise RuntimeError(
            f"trials.jsonl records {n_trials} trials but {ledger_dir} holds "
            f"{n_ledgers} distinct configs. Someone ran backtests without "
            "logging them; refusing to build."
        )
    return n_trials
