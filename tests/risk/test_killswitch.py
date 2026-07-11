"""Inject each of the six halt conditions; each must trip with the offending
value in str(reason)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from twenty.risk.killswitch import (
    BrokerState,
    HaltCause,
    PortfolioState,
    should_halt,
)

NOW = datetime(2026, 7, 10, 15, 0, tzinfo=UTC)


def healthy_portfolio() -> PortfolioState:
    return PortfolioState(
        equity=Decimal("20.50"),
        day_start_equity=Decimal("20.40"),
        high_water_mark=Decimal("21.00"),
        positions={"SPY": Decimal("0.0150")},
        cash=Decimal("11.00"),
    )


def healthy_broker() -> BrokerState:
    return BrokerState(
        positions={"SPY": Decimal("0.0150")},
        cash=Decimal("11.00"),
        last_tick_at=NOW - timedelta(seconds=5),
        is_rth=True,
        consecutive_rejections=0,
    )


def test_healthy_state_does_not_halt(tmp_path: Path) -> None:
    reason = should_halt(
        healthy_portfolio(), healthy_broker(), halt_file=tmp_path / "HALT", now=NOW
    )
    assert reason is None


def test_halt_file(tmp_path: Path) -> None:
    halt = tmp_path / "HALT"
    halt.write_text("manual stop for gateway upgrade")
    reason = should_halt(healthy_portfolio(), healthy_broker(), halt_file=halt, now=NOW)
    assert reason is not None
    assert reason.cause is HaltCause.HALT_FILE
    assert "manual stop for gateway upgrade" in str(reason)


def test_cash_reconciliation_mismatch(tmp_path: Path) -> None:
    broker = BrokerState(
        positions={"SPY": Decimal("0.0150")},
        cash=Decimal("10.50"),
        last_tick_at=NOW,
        is_rth=True,
        consecutive_rejections=0,
    )
    reason = should_halt(
        healthy_portfolio(), broker, halt_file=tmp_path / "HALT", now=NOW
    )
    assert reason is not None
    assert reason.cause is HaltCause.RECONCILIATION_MISMATCH
    assert "0.50" in str(reason)


def test_position_reconciliation_mismatch(tmp_path: Path) -> None:
    broker = BrokerState(
        positions={"SPY": Decimal("0.0300")},
        cash=Decimal("11.00"),
        last_tick_at=NOW,
        is_rth=True,
        consecutive_rejections=0,
    )
    reason = should_halt(
        healthy_portfolio(), broker, halt_file=tmp_path / "HALT", now=NOW
    )
    assert reason is not None
    assert reason.cause is HaltCause.RECONCILIATION_MISMATCH
    assert "SPY" in str(reason)
    assert "0.0300" in str(reason)


def test_stale_market_data(tmp_path: Path) -> None:
    broker = BrokerState(
        positions={"SPY": Decimal("0.0150")},
        cash=Decimal("11.00"),
        last_tick_at=NOW - timedelta(seconds=400),
        is_rth=True,
        consecutive_rejections=0,
    )
    reason = should_halt(
        healthy_portfolio(), broker, halt_file=tmp_path / "HALT", now=NOW
    )
    assert reason is not None
    assert reason.cause is HaltCause.STALE_MARKET_DATA
    assert "400" in str(reason)


def test_staleness_ignored_outside_rth(tmp_path: Path) -> None:
    broker = BrokerState(
        positions={"SPY": Decimal("0.0150")},
        cash=Decimal("11.00"),
        last_tick_at=NOW - timedelta(hours=10),
        is_rth=False,
        consecutive_rejections=0,
    )
    assert should_halt(
        healthy_portfolio(), broker, halt_file=tmp_path / "HALT", now=NOW
    ) is None


def test_daily_loss(tmp_path: Path) -> None:
    portfolio = PortfolioState(
        equity=Decimal("17.00"),
        day_start_equity=Decimal("20.00"),
        high_water_mark=Decimal("21.00"),
        positions={"SPY": Decimal("0.0150")},
        cash=Decimal("11.00"),
    )
    reason = should_halt(portfolio, healthy_broker(), halt_file=tmp_path / "HALT", now=NOW)
    assert reason is not None
    assert reason.cause is HaltCause.DAILY_LOSS
    assert "15.0%" in str(reason)


def test_drawdown(tmp_path: Path) -> None:
    portfolio = PortfolioState(
        equity=Decimal("13.00"),
        day_start_equity=Decimal("13.10"),
        high_water_mark=Decimal("21.00"),
        positions={"SPY": Decimal("0.0150")},
        cash=Decimal("13.00"),
    )
    broker = BrokerState(
        positions={"SPY": Decimal("0.0150")},
        cash=Decimal("13.00"),
        last_tick_at=NOW,
        is_rth=True,
        consecutive_rejections=0,
    )
    reason = should_halt(portfolio, broker, halt_file=tmp_path / "HALT", now=NOW)
    assert reason is not None
    assert reason.cause is HaltCause.DRAWDOWN
    assert "38.1%" in str(reason)


def test_consecutive_rejections(tmp_path: Path) -> None:
    broker = BrokerState(
        positions={"SPY": Decimal("0.0150")},
        cash=Decimal("11.00"),
        last_tick_at=NOW,
        is_rth=True,
        consecutive_rejections=3,
    )
    reason = should_halt(
        healthy_portfolio(), broker, halt_file=tmp_path / "HALT", now=NOW
    )
    assert reason is not None
    assert reason.cause is HaltCause.CONSECUTIVE_REJECTIONS
    assert "3" in str(reason)


def test_halt_file_beats_everything(tmp_path: Path) -> None:
    """Order matters: the file check must fire even when later inputs are
    garbage, because it must work when other subsystems are down."""
    halt = tmp_path / "HALT"
    halt.write_text("stop")
    broker = BrokerState(
        positions={},
        cash=Decimal("-999"),
        last_tick_at=None,
        is_rth=True,
        consecutive_rejections=99,
    )
    reason = should_halt(healthy_portfolio(), broker, halt_file=halt, now=NOW)
    assert reason is not None
    assert reason.cause is HaltCause.HALT_FILE
