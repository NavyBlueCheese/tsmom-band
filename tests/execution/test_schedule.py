from __future__ import annotations

from datetime import UTC, datetime

from twenty.execution.schedule import next_trigger, quarter_end_sessions, trigger_time_utc


def test_trigger_utc_offset_differs_across_dst() -> None:
    """15:45 New York is 19:45 UTC in summer and 20:45 UTC in winter. If a
    UTC offset were hardcoded, one of these would fail."""
    summer = trigger_time_utc(datetime(2026, 6, 30, tzinfo=UTC))
    winter = trigger_time_utc(datetime(2026, 12, 31, tzinfo=UTC))
    assert (summer.hour, summer.minute) == (19, 45)
    assert (winter.hour, winter.minute) == (20, 45)


def test_quarter_end_sessions_2026() -> None:
    sessions = quarter_end_sessions(2026)
    assert len(sessions) == 4
    assert all(s.weekday() < 5 for s in sessions)


def test_next_trigger_strictly_after_now() -> None:
    now = datetime(2026, 7, 10, 12, 0, tzinfo=UTC)
    trigger = next_trigger(now)
    assert trigger > now
    # Next quarter end after July 10 is the end of September.
    assert trigger.month == 9


def test_force_weekly_gives_trigger_within_eight_days() -> None:
    now = datetime(2026, 7, 10, 0, 0, tzinfo=UTC)
    trigger = next_trigger(now, force_weekly=True)
    assert trigger > now
    assert (trigger - now).days <= 8
