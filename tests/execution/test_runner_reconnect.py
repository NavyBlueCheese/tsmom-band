"""The daily Gateway logout must be survivable: reconnect with patience."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from twenty.execution.runner import reconnect_with_patience


def test_reconnects_after_transient_failures(tmp_path: Path) -> None:
    attempts: list[int] = []
    slept: list[float] = []

    def connect_fn() -> Any:
        attempts.append(1)
        if len(attempts) < 3:
            raise ConnectionError("gateway restarting")
        return "ib"

    result = reconnect_with_patience(
        connect_fn,
        max_attempts=10,
        interval_s=60.0,
        sleep_fn=slept.append,
        halt_file=tmp_path / "HALT",
    )
    assert result == "ib"
    assert len(attempts) == 3
    assert slept == [60.0, 60.0]
    assert not (tmp_path / "HALT").exists()


def test_halts_after_patience_exhausted(tmp_path: Path) -> None:
    halt = tmp_path / "HALT"

    def connect_fn() -> Any:
        raise ConnectionError("gateway gone")

    with pytest.raises(SystemExit):
        reconnect_with_patience(
            connect_fn,
            max_attempts=3,
            interval_s=60.0,
            sleep_fn=lambda _: None,
            halt_file=halt,
        )
    assert halt.exists()
    assert "unreachable" in halt.read_text()


def test_respects_halt_file_during_retries(tmp_path: Path) -> None:
    halt = tmp_path / "HALT"
    calls: list[int] = []

    def connect_fn() -> Any:
        calls.append(1)
        halt.write_text("human said stop")
        raise ConnectionError("down")

    with pytest.raises(SystemExit):
        reconnect_with_patience(
            connect_fn,
            max_attempts=10,
            interval_s=60.0,
            sleep_fn=lambda _: None,
            halt_file=halt,
        )
    assert len(calls) == 1  # stopped at the next check, no retry storm
