from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo
import math
import pandas as pd


def _parse_hhmm(value: str) -> time:
    h, m = value.strip().split(":", 1)
    return time(int(h), int(m))


def estimate_resolution_seconds(df: pd.DataFrame) -> float:
    if df.empty or "timestamp" not in df.columns:
        return 60.0
    diff = df["timestamp"].astype(float).sort_values().diff().dropna()
    diff = diff[(diff > 0) & (diff <= 3600)]
    if diff.empty:
        return 60.0
    return float(diff.median())


def session_bounds_utc(session_date: date, timezone_name: str, start_hhmm: str, end_hhmm: str) -> tuple[float, float]:
    tz = ZoneInfo(timezone_name)
    start_local = datetime.combine(session_date, _parse_hhmm(start_hhmm), tzinfo=tz)
    end_local = datetime.combine(session_date, _parse_hhmm(end_hhmm), tzinfo=tz)
    if end_local <= start_local:
        end_local += timedelta(days=1)
    return start_local.timestamp(), end_local.timestamp()


def _iso_utc_from_ts(ts: float) -> str:
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat().replace("+00:00", "Z")


def extract_rth_session(
    raw_df: pd.DataFrame,
    session_date: date,
    timezone_name: str = "America/New_York",
    start_hhmm: str = "09:30",
    end_hhmm: str = "16:00",
    min_coverage: float = 0.80,
) -> dict | None:
    """Return RTH stats only when the session looks substantially complete.

    The session is [start, end). Validation is resolution-aware so both M1 and
    coarser historical files can be used. A shortened/partial session will be
    rejected because it does not cover the expected end of RTH.
    """
    if raw_df.empty:
        return None
    if session_date.weekday() >= 5:
        return None

    start_ts, end_ts = session_bounds_utc(session_date, timezone_name, start_hhmm, end_hhmm)
    sub = raw_df[(raw_df["timestamp"] >= start_ts) & (raw_df["timestamp"] < end_ts)].copy()
    if sub.empty:
        return None

    resolution = max(1.0, estimate_resolution_seconds(raw_df))
    session_seconds = max(1.0, end_ts - start_ts)
    expected_count = max(1, int(math.floor(session_seconds / resolution + 1e-9)))
    required_count = max(1, int(math.floor(expected_count * float(min_coverage))))

    first_ts = float(sub["timestamp"].min())
    last_ts = float(sub["timestamp"].max())
    # Allow up to two bars of tolerance at either edge, while still rejecting
    # holidays / heavily truncated sessions / early closes.
    edge_tolerance = max(resolution * 2.0, 120.0)
    if len(sub) < required_count:
        return None
    if first_ts > start_ts + edge_tolerance:
        return None
    if last_ts < end_ts - edge_tolerance - resolution:
        return None

    high_idx = sub["high"].astype(float).idxmax()
    low_idx = sub["low"].astype(float).idxmin()
    high_row = sub.loc[high_idx]
    low_row = sub.loc[low_idx]

    return {
        "session_date": session_date.isoformat(),
        "session_timezone": timezone_name,
        "session_start": start_hhmm,
        "session_end": end_hhmm,
        "high": float(high_row["high"]),
        "low": float(low_row["low"]),
        "high_time": _iso_utc_from_ts(float(high_row["timestamp"])),
        "low_time": _iso_utc_from_ts(float(low_row["timestamp"])),
        "source_resolution_seconds": float(resolution),
        "bar_count": int(len(sub)),
    }


def find_previous_valid_rth(
    raw_df: pd.DataFrame,
    case_date: date,
    timezone_name: str = "America/New_York",
    start_hhmm: str = "09:30",
    end_hhmm: str = "16:00",
    lookback_days: int = 14,
    min_coverage: float = 0.80,
) -> dict | None:
    """Find the most recent complete RTH session strictly before case_date."""
    for days_back in range(1, max(1, int(lookback_days)) + 1):
        candidate = case_date - timedelta(days=days_back)
        if candidate.weekday() >= 5:
            continue
        result = extract_rth_session(
            raw_df,
            candidate,
            timezone_name=timezone_name,
            start_hhmm=start_hhmm,
            end_hhmm=end_hhmm,
            min_coverage=min_coverage,
        )
        if result is not None:
            return result
    return None
