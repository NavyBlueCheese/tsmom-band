"""Execution tests. The ib_async surface is mocked entirely."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from twenty.backtest.types import PortfolioView, ProposedOrder, Side
from twenty.execution.connection import (
    GATEWAY_LIVE_PORT,
    GATEWAY_PAPER_PORT,
    TWS_PAPER_PORT,
    IBSettings,
    PortSafetyError,
    assert_port_consistent,
)
from twenty.execution.journal import Journal, OrderState
from twenty.execution.reconcile import ReconciliationError, reconcile
from twenty.risk.checks import PreTradeRiskCheck


def order(notional: str = "5.00", side: Side = Side.BUY) -> ProposedOrder:
    return ProposedOrder(symbol="SPY", side=side, notional=Decimal(notional))


# --- port safety ---------------------------------------------------------------


def test_paper_flag_with_live_port_refuses() -> None:
    settings = IBSettings(port=GATEWAY_LIVE_PORT, is_paper=True)
    with pytest.raises(PortSafetyError):
        assert_port_consistent(settings)


def test_live_flag_with_paper_port_refuses() -> None:
    for port in (GATEWAY_PAPER_PORT, TWS_PAPER_PORT):
        settings = IBSettings(port=port, is_paper=False)
        with pytest.raises(PortSafetyError):
            assert_port_consistent(settings)


def test_consistent_ports_pass() -> None:
    assert_port_consistent(IBSettings(port=GATEWAY_PAPER_PORT, is_paper=True))
    assert_port_consistent(IBSettings(port=GATEWAY_LIVE_PORT, is_paper=False))


# --- journal: the anti-double-position property ---------------------------------


def test_journal_written_before_transmission(tmp_path: Path) -> None:
    journal = Journal(tmp_path / "j.sqlite")
    o = order()
    journal.record_proposed(o)
    assert journal.state_of(o.client_id) is OrderState.PENDING


def test_restart_mid_order_no_duplicate(tmp_path: Path) -> None:
    """Journal an order, transmit it, kill the process (drop the connection),
    restart, and assert the same UUID is never transmitted again."""
    path = tmp_path / "j.sqlite"
    journal = Journal(path)
    o = order()
    journal.record_proposed(o)
    journal.set_state(o.client_id, OrderState.TRANSMITTED, broker_order_id=101)
    journal.close()  # process dies here

    restarted = Journal(path)  # restart
    # Broker still shows order 101 open.
    live = restarted.reconcile_on_startup(open_broker_ids={101}, filled_broker_ids=set())
    assert live == [o.client_id]
    assert restarted.may_transmit(o) is False


def test_restart_after_fill_marks_filled(tmp_path: Path) -> None:
    path = tmp_path / "j.sqlite"
    journal = Journal(path)
    o = order()
    journal.record_proposed(o)
    journal.set_state(o.client_id, OrderState.TRANSMITTED, broker_order_id=7)
    journal.close()

    restarted = Journal(path)
    live = restarted.reconcile_on_startup(open_broker_ids=set(), filled_broker_ids={7})
    assert live == []
    assert restarted.state_of(o.client_id) is OrderState.FILLED
    assert restarted.may_transmit(o) is False


def test_error_10349_scenario_no_retry(tmp_path: Path) -> None:
    """The 10349 class of failure: the client sees an error while the order
    stays live at the broker. The journal keeps the order TRANSMITTED, so a
    retry of the same UUID is refused and the position cannot double."""
    journal = Journal(tmp_path / "j.sqlite")
    o = order()
    journal.record_proposed(o)
    journal.set_state(o.client_id, OrderState.TRANSMITTED, broker_order_id=55)
    # Client-side error arrives; nothing marks the order terminal.
    assert journal.state_of(o.client_id) is OrderState.TRANSMITTED
    assert journal.may_transmit(o) is False
    # Only after reconciliation shows it truly gone may a NEW order (new UUID)
    # be considered — the old UUID stays refused forever.
    assert journal.may_transmit(order()) is True  # different UUID, fresh order


def test_pending_never_transmitted_is_cancelled_on_restart(tmp_path: Path) -> None:
    path = tmp_path / "j.sqlite"
    journal = Journal(path)
    o = order()
    journal.record_proposed(o)  # journalled, process dies before transmit
    journal.close()

    restarted = Journal(path)
    live = restarted.reconcile_on_startup(open_broker_ids=set(), filled_broker_ids=set())
    assert live == []
    assert restarted.state_of(o.client_id) is OrderState.CANCELLED


# --- reconcile ------------------------------------------------------------------


def test_reconcile_mismatch_writes_halt_and_raises(tmp_path: Path) -> None:
    halt = tmp_path / "HALT"
    with pytest.raises(ReconciliationError):
        reconcile(
            local_positions={"SPY": Decimal("0.01")},
            local_cash=Decimal("10.00"),
            broker_positions={"SPY": Decimal("0.02")},
            broker_cash=Decimal("10.00"),
            halt_file=halt,
        )
    assert halt.exists()
    assert "SPY" in halt.read_text()


def test_reconcile_within_tolerance_passes(tmp_path: Path) -> None:
    reconcile(
        local_positions={"SPY": Decimal("0.0100")},
        local_cash=Decimal("10.00"),
        broker_positions={"SPY": Decimal("0.0100")},
        broker_cash=Decimal("10.01"),
        halt_file=tmp_path / "HALT",
    )
    assert not (tmp_path / "HALT").exists()


# --- sizing floor ----------------------------------------------------------------


def test_one_eighty_delta_produces_no_order() -> None:
    """A $1.80 delta is below MIN_ORDER_USD and must be dropped by the risk
    check even if a strategy emitted it."""
    view = PortfolioView(positions={}, cash=Decimal("20.00"))
    out = PreTradeRiskCheck().check([order(notional="1.80")], view)
    assert out == []
