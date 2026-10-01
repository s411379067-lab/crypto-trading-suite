from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo
import math
import pandas as pd


DEFAULT_MIN_COVERAGE = 0.95
DEFAULT_MAX_TAIL_GAP_MINUTES = 15.0


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
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    max_tail_gap_minutes: float = DEFAULT_MAX_TAIL_GAP_MINUTES,
) -> dict | None:
    """Return stats for substantially complete RTH sessions.

    The theoretical session remains ``start_hhmm`` -> ``end_hhmm``.  Some CFD
    brokers enter a short daily maintenance break before the cash-session end
    during part of the year (for NAS100 this can leave the last available M1
    bar at 15:49 New York time).  Such a session is still accepted when:

    * bar coverage is at least ``min_coverage``;
    * the session starts near the theoretical open; and
    * the unquoted tail after the final bar is no longer than
      ``max_tail_gap_minutes``.

    This keeps genuine early-close / badly truncated sessions rejected while
    avoiding false negatives caused by a short broker maintenance window.
    """
    if raw_df.empty or session_date.weekday() >= 5:
        return None

    start_ts, end_ts = session_bounds_utc(session_date, timezone_name, start_hhmm, end_hhmm)
    sub = raw_df[(raw_df["timestamp"] >= start_ts) & (raw_df["timestamp"] < end_ts)].copy()
    if sub.empty:
        return None

    resolution = max(1.0, estimate_resolution_seconds(raw_df))
    session_seconds = max(1.0, end_ts - start_ts)
    expected_count = max(1, int(math.floor(session_seconds / resolution + 1e-9)))
    required_count = max(1, int(math.ceil(expected_count * float(min_coverage) - 1e-12)))

    first_ts = float(sub["timestamp"].min())
    last_ts = float(sub["timestamp"].max())
    start_tolerance = max(resolution * 2.0, 120.0)
    # The last timestamp is the *start* of the final bar.  Measure the truly
    # unquoted tail from the end of that bar to the theoretical RTH end.
    tail_gap_seconds = max(0.0, end_ts - (last_ts + resolution))
    max_tail_gap_seconds = max(0.0, float(max_tail_gap_minutes) * 60.0)
    if len(sub) < required_count:
        return None
    if first_ts > start_ts + start_tolerance:
        return None
    if tail_gap_seconds > max_tail_gap_seconds + 1e-9:
        return None

    coverage_ratio = float(len(sub)) / float(expected_count)

    ordered = sub.sort_values("timestamp")
    open_row = ordered.iloc[0]
    close_row = ordered.iloc[-1]
    high_idx = sub["high"].astype(float).idxmax()
    low_idx = sub["low"].astype(float).idxmin()
    high_row = sub.loc[high_idx]
    low_row = sub.loc[low_idx]

    return {
        "session_date": session_date.isoformat(),
        "session_timezone": timezone_name,
        "session_start": start_hhmm,
        "session_end": end_hhmm,
        "min_coverage": float(min_coverage),
        "max_tail_gap_minutes": float(max_tail_gap_minutes),
        "open": float(open_row["open"]),
        "high": float(high_row["high"]),
        "low": float(low_row["low"]),
        "close": float(close_row["close"]),
        "open_time": _iso_utc_from_ts(float(open_row["timestamp"])),
        "high_time": _iso_utc_from_ts(float(high_row["timestamp"])),
        "low_time": _iso_utc_from_ts(float(low_row["timestamp"])),
        "close_time": _iso_utc_from_ts(float(close_row["timestamp"])),
        "source_resolution_seconds": float(resolution),
        "bar_count": int(len(sub)),
        "expected_bar_count": int(expected_count),
        "coverage_ratio": float(coverage_ratio),
        "tail_gap_minutes": float(tail_gap_seconds / 60.0),
    }


def previous_rth_is_current(
    rth: dict | None,
    *,
    calculator_version: str,
    timezone_name: str,
    start_hhmm: str,
    end_hhmm: str,
    data_source_id: str = "",
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    max_tail_gap_minutes: float = DEFAULT_MAX_TAIL_GAP_MINUTES,
) -> bool:
    if not isinstance(rth, dict):
        return False
    required = (
        "session_date", "session_timezone", "session_start", "session_end",
        "min_coverage", "max_tail_gap_minutes",
        "high", "low", "close", "high_time", "low_time", "close_time",
        "calculator_version",
    )
    if any(rth.get(key) is None or rth.get(key) == "" for key in required):
        return False
    if str(rth.get("calculator_version")) != str(calculator_version):
        return False
    if str(rth.get("session_timezone")) != str(timezone_name):
        return False
    if str(rth.get("session_start")) != str(start_hhmm):
        return False
    if str(rth.get("session_end")) != str(end_hhmm):
        return False
    try:
        if abs(float(rth.get("min_coverage")) - float(min_coverage)) > 1e-12:
            return False
        if abs(float(rth.get("max_tail_gap_minutes")) - float(max_tail_gap_minutes)) > 1e-12:
            return False
    except Exception:
        return False
    if data_source_id and str(rth.get("data_source_id", "")) != str(data_source_id):
        return False
    try:
        float(rth["high"]); float(rth["low"]); float(rth["close"])
    except Exception:
        return False
    return True


def find_previous_valid_rth(
    raw_df: pd.DataFrame,
    case_date: date,
    timezone_name: str = "America/New_York",
    start_hhmm: str = "09:30",
    end_hhmm: str = "16:00",
    lookback_days: int = 14,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    max_tail_gap_minutes: float = DEFAULT_MAX_TAIL_GAP_MINUTES,
) -> dict | None:
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
            max_tail_gap_minutes=max_tail_gap_minutes,
        )
        if result is not None:
            return result
    return None


def collect_previous_valid_rth_sessions(
    raw_df: pd.DataFrame,
    case_date: date,
    *,
    session_count: int = 20,
    timezone_name: str = "America/New_York",
    start_hhmm: str = "09:30",
    end_hhmm: str = "16:00",
    lookback_days: int = 60,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    max_tail_gap_minutes: float = DEFAULT_MAX_TAIL_GAP_MINUTES,
) -> list[dict]:
    target = max(1, int(session_count))
    found: list[dict] = []
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
            max_tail_gap_minutes=max_tail_gap_minutes,
        )
        if result is None:
            continue
        found.append(result)
        if len(found) >= target:
            break
    return list(reversed(found))


def calculate_intraday_volatility(
    raw_df: pd.DataFrame,
    case_date: date,
    *,
    session_count: int = 20,
    timezone_name: str = "America/New_York",
    start_hhmm: str = "09:30",
    end_hhmm: str = "16:00",
    lookback_days: int = 60,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    max_tail_gap_minutes: float = DEFAULT_MAX_TAIL_GAP_MINUTES,
) -> dict | None:
    """Recent RTH high-low range stats. Std is sample std (ddof=1)."""
    target = max(2, int(session_count))
    sessions = collect_previous_valid_rth_sessions(
        raw_df,
        case_date,
        session_count=target,
        timezone_name=timezone_name,
        start_hhmm=start_hhmm,
        end_hhmm=end_hhmm,
        lookback_days=lookback_days,
        min_coverage=min_coverage,
        max_tail_gap_minutes=max_tail_gap_minutes,
    )
    if len(sessions) < target:
        return None

    rows: list[dict] = []
    for session in sessions:
        open_price = float(session["open"])
        range_points = float(session["high"]) - float(session["low"])
        if open_price == 0:
            return None
        range_pct = range_points / open_price * 100.0
        rows.append({
            "date": session["session_date"],
            "open": open_price,
            "high": float(session["high"]),
            "low": float(session["low"]),
            "close": float(session["close"]),
            "range_points": float(range_points),
            "range_pct": float(range_pct),
            "bar_count": int(session.get("bar_count", 0)),
            "expected_bar_count": int(session.get("expected_bar_count", 0)),
            "coverage_ratio": float(session.get("coverage_ratio", 0.0)),
            "tail_gap_minutes": float(session.get("tail_gap_minutes", 0.0)),
        })

    points = pd.Series([row["range_points"] for row in rows], dtype="float64")
    pcts = pd.Series([row["range_pct"] for row in rows], dtype="float64")
    return {
        "session_timezone": timezone_name,
        "session_start": start_hhmm,
        "session_end": end_hhmm,
        "min_coverage": float(min_coverage),
        "max_tail_gap_minutes": float(max_tail_gap_minutes),
        "lookback_sessions": target,
        "observations": len(rows),
        "range_definition": "high_minus_low",
        "range_pct_definition": "(high-low)/rth_open*100",
        "std_method": "sample_ddof_1",
        "median_range_points": float(points.median()),
        "std_range_points": float(points.std(ddof=1)),
        "median_range_pct": float(pcts.median()),
        "std_range_pct": float(pcts.std(ddof=1)),
        "sessions": rows,
    }


def intraday_volatility_is_current(
    payload: dict | None,
    *,
    calculator_version: str,
    session_count: int,
    timezone_name: str,
    start_hhmm: str,
    end_hhmm: str,
    data_source_id: str = "",
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    max_tail_gap_minutes: float = DEFAULT_MAX_TAIL_GAP_MINUTES,
) -> bool:
    if not isinstance(payload, dict):
        return False
    required = (
        "session_timezone", "session_start", "session_end", "min_coverage", "max_tail_gap_minutes",
        "lookback_sessions", "observations", "range_definition", "range_pct_definition", "std_method",
        "median_range_points", "std_range_points", "median_range_pct", "std_range_pct",
        "sessions", "calculator_version",
    )
    if any(payload.get(key) is None or payload.get(key) == "" for key in required):
        return False
    if str(payload.get("calculator_version")) != str(calculator_version):
        return False
    if int(payload.get("lookback_sessions", 0)) != int(session_count):
        return False
    if int(payload.get("observations", 0)) != int(session_count):
        return False
    if str(payload.get("session_timezone")) != str(timezone_name):
        return False
    if str(payload.get("session_start")) != str(start_hhmm):
        return False
    if str(payload.get("session_end")) != str(end_hhmm):
        return False
    try:
        if abs(float(payload.get("min_coverage")) - float(min_coverage)) > 1e-12:
            return False
        if abs(float(payload.get("max_tail_gap_minutes")) - float(max_tail_gap_minutes)) > 1e-12:
            return False
    except Exception:
        return False
    if str(payload.get("std_method")) != "sample_ddof_1":
        return False
    if data_source_id and str(payload.get("data_source_id", "")) != str(data_source_id):
        return False
    sessions = payload.get("sessions")
    if not isinstance(sessions, list) or len(sessions) != int(session_count):
        return False
    try:
        float(payload["median_range_points"])
        float(payload["std_range_points"])
        float(payload["median_range_pct"])
        float(payload["std_range_pct"])
    except Exception:
        return False
    return True


def summarize_intraday_volatility_payload(payload: dict | None, session_count: int) -> dict | None:
    """Summarize the most recent N enriched sessions without touching raw data.

    This is intentionally an Analyzer-level summary over the Enricher's stored
    daily ``range_pct`` values.  It never redefines sessions or recalculates OHLC.
    Sample standard deviation (ddof=1) matches the Enricher definition.
    """
    if not isinstance(payload, dict):
        return None
    try:
        n = int(session_count)
    except Exception:
        return None
    if n < 2 or n > 20:
        return None
    sessions = payload.get("sessions")
    if not isinstance(sessions, list) or len(sessions) < n:
        return None
    recent = sessions[-n:]
    try:
        values = pd.Series([float(row["range_pct"]) for row in recent], dtype="float64")
    except Exception:
        return None
    if len(values) != n or values.isna().any():
        return None
    median = float(values.median())
    std = float(values.std(ddof=1))
    return {
        "session_count": n,
        "median_range_pct": median,
        "std_range_pct": std,
        "std_1x_range_pct": std,
        "std_2x_range_pct": std * 2.0,
        "std_3x_range_pct": std * 3.0,
        "start_date": str(recent[0].get("date", "")),
        "end_date": str(recent[-1].get("date", "")),
    }
