from datetime import date
import pandas as pd

from shared_core.rth import extract_rth_session, find_previous_valid_rth


def make_session(day: str):
    dt = pd.date_range(f"{day} 13:30:00+00:00", periods=390, freq="1min")  # 09:30 NY during EDT
    return pd.DataFrame({
        "dt_utc": dt,
        "timestamp": dt.astype("int64") / 1_000_000_000,
        "open": range(390),
        "high": [1000 + i for i in range(390)],
        "low": [-i for i in range(390)],
        "close": range(390),
    })


def test_extract_complete_rth():
    df = make_session("2026-09-18")
    out = extract_rth_session(df, date(2026, 9, 18))
    assert out is not None
    assert out["session_date"] == "2026-09-18"
    assert out["high"] == 1389.0
    assert out["low"] == -389.0


def test_monday_finds_friday_not_weekend():
    df = make_session("2026-09-18")
    out = find_previous_valid_rth(df, date(2026, 9, 21))
    assert out is not None
    assert out["session_date"] == "2026-09-18"


def test_partial_session_is_rejected():
    df = make_session("2026-09-18").iloc[:120].copy()
    out = extract_rth_session(df, date(2026, 9, 18))
    assert out is None


def test_extract_complete_rth_includes_close():
    df = make_session("2026-09-18")
    out = extract_rth_session(df, date(2026, 9, 18))
    assert out is not None
    assert out["close"] == 389.0
    assert out["close_time"].startswith("2026-09-18T19:59:00")


def test_previous_rth_completeness_upgrades_old_payload():
    from shared_core.rth import previous_rth_is_current

    old_payload = {
        "session_date": "2026-09-18",
        "session_timezone": "America/New_York",
        "session_start": "09:30",
        "session_end": "16:00",
        "high": 100.0,
        "low": 90.0,
        "high_time": "2026-09-18T15:00:00Z",
        "low_time": "2026-09-18T16:00:00Z",
        "calculator_version": "1.0",
        "data_source_id": "nas100-primary",
    }
    assert not previous_rth_is_current(
        old_payload,
        calculator_version="1.1",
        timezone_name="America/New_York",
        start_hhmm="09:30",
        end_hhmm="16:00",
        data_source_id="nas100-primary",
    )

    new_payload = dict(old_payload)
    new_payload.update({
        "close": 95.0,
        "close_time": "2026-09-18T19:59:00Z",
        "calculator_version": "1.1",
    })
    assert previous_rth_is_current(
        new_payload,
        calculator_version="1.1",
        timezone_name="America/New_York",
        start_hhmm="09:30",
        end_hhmm="16:00",
        data_source_id="nas100-primary",
    )
