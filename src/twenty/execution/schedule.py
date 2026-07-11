"""Scheduling. Everything in UTC internally; the exchange calendar owns
session boundaries, holidays, and early closes.

The trigger is 15:45 America/New_York on the last session of each calendar
quarter. Never hardcode a UTC offset: the operator is in KST and US DST will
silently break naive arithmetic twice a year. zoneinfo carries the DST rules;
the conversion below is always through the named zone.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import exchange_calendars as xcals

NY = ZoneInfo("America/New_York")
TRIGGER_HOUR = 15
TRIGGER_MINUTE = 45


def _calendar(start_year: int, end_year: int) -> xcals.ExchangeCalendar:
    return xcals.get_calendar(
        "XNYS", start=f"{start_year - 1}-12-01", end=f"{end_year + 1}-01-31"
    )


def quarter_end_sessions(year: int) -> list[datetime]:
    """The last XNYS session of each quarter of ``year``, as dates at UTC
    midnight."""
    cal = _calendar(year, year)
    out: list[datetime] = []
    for month, day in ((3, 31), (6, 30), (9, 30), (12, 31)):
        session = cal.date_to_session(f"{year}-{month:02d}-{day:02d}", direction="previous")
        out.append(datetime(session.year, session.month, session.day, tzinfo=UTC))
    return out


def trigger_time_utc(session_date: datetime) -> datetime:
    """15:45 America/New_York on the given session, expressed in UTC. The
    offset differs by an hour between winter and summer; zoneinfo, not
    arithmetic, decides which applies."""
    local = datetime(
        session_date.year,
        session_date.month,
        session_date.day,
        TRIGGER_HOUR,
        TRIGGER_MINUTE,
        tzinfo=NY,
    )
    return local.astimezone(UTC)


def next_trigger(
    now: datetime, force_weekly: bool = False
) -> datetime:
    """The next rebalance trigger strictly after ``now`` (UTC).

    ``force_weekly`` (paper testing only) overrides the quarter-end check —
    and only that check: the trigger becomes 15:45 New York on every Friday
    session (or the last session of the week when Friday is a holiday).
    """
    now = now.astimezone(UTC)
    if force_weekly:
        cal = _calendar(now.year, now.year + 1)
        probe = now.date()
        for _ in range(400):
            session_str = probe.isoformat()
            if cal.is_session(session_str):
                session = cal.date_to_session(session_str)
                is_last_of_week = cal.next_session(session).weekday() < session.weekday()
                if is_last_of_week:
                    trigger = trigger_time_utc(
                        datetime(probe.year, probe.month, probe.day, tzinfo=UTC)
                    )
                    if trigger > now:
                        return trigger
            probe = probe + timedelta(days=1)
        raise RuntimeError("No weekly trigger found within 400 days")
    candidates: list[datetime] = []
    for year in (now.year, now.year + 1):
        candidates.extend(quarter_end_sessions(year))
    for session in sorted(candidates):
        trigger = trigger_time_utc(session)
        if trigger > now:
            return trigger
    raise RuntimeError("No quarter-end trigger found; calendar bounds too narrow")
