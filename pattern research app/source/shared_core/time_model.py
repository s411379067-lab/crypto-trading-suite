from __future__ import annotations

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo


def build_case_time_range(
    research_date: date,
    data_start_time: time,
    replay_start_time: time,
    default_end_time: time,
    timezone_name: str,
) -> dict[str, str]:
    """Build chronological A/B/C timestamps in one semantic case timezone.

    A starts on research_date. B is placed on the same date unless its local
    clock time would be earlier than A; in that case B moves to the next day.
    C is placed on B's local date unless its local clock time would be earlier
    than B; in that case C moves to the following day.

    This keeps A <= B <= C while preserving local wall-clock semantics and DST.
    """
    tz = ZoneInfo(timezone_name)

    a = datetime.combine(research_date, data_start_time, tzinfo=tz)

    b_date = research_date
    b = datetime.combine(b_date, replay_start_time, tzinfo=tz)
    if b < a:
        b_date = b_date + timedelta(days=1)
        b = datetime.combine(b_date, replay_start_time, tzinfo=tz)

    c_date = b_date
    c = datetime.combine(c_date, default_end_time, tzinfo=tz)
    if c < b:
        c_date = c_date + timedelta(days=1)
        c = datetime.combine(c_date, default_end_time, tzinfo=tz)

    return {
        "data_start": a.isoformat(),
        "replay_start": b.isoformat(),
        "default_end": c.isoformat(),
    }
