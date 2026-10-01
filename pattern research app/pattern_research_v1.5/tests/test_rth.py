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
