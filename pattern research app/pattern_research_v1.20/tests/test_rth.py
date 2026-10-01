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


def _make_flat_session(day: str, open_price: float, range_points: float):
    dt = pd.date_range(f"{day} 13:30:00+00:00", periods=390, freq="1min")
    highs = [open_price + range_points * (i / 389.0) for i in range(390)]
    lows = [open_price for _ in range(390)]
    closes = [open_price + range_points * 0.5 for _ in range(390)]
    return pd.DataFrame({
        "dt_utc": dt,
        "timestamp": dt.astype("int64") / 1_000_000_000,
        "open": [open_price for _ in range(390)],
        "high": highs,
        "low": lows,
        "close": closes,
    })


def test_20d_intraday_volatility_median_and_sample_std():
    from shared_core.rth import calculate_intraday_volatility

    days = pd.bdate_range(end="2026-09-29", periods=20)
    frames = []
    ranges = []
    for i, day in enumerate(days, start=1):
        r = float(i * 10)
        ranges.append(r)
        frames.append(_make_flat_session(day.strftime("%Y-%m-%d"), 1000.0, r))
    df = pd.concat(frames, ignore_index=True)

    out = calculate_intraday_volatility(df, date(2026, 9, 30), session_count=20)
    assert out is not None
    assert out["observations"] == 20
    assert out["median_range_points"] == 105.0
    expected_std = pd.Series(ranges, dtype="float64").std(ddof=1)
    assert abs(out["std_range_points"] - expected_std) < 1e-9
    assert abs(out["median_range_pct"] - 10.5) < 1e-9
    assert abs(out["std_range_pct"] - expected_std / 10.0) < 1e-9
    assert out["std_method"] == "sample_ddof_1"
    assert len(out["sessions"]) == 20


def test_20d_intraday_volatility_requires_full_20_sessions():
    from shared_core.rth import calculate_intraday_volatility

    days = pd.bdate_range(end="2026-09-29", periods=19)
    df = pd.concat([
        _make_flat_session(day.strftime("%Y-%m-%d"), 1000.0, 100.0)
        for day in days
    ], ignore_index=True)
    assert calculate_intraday_volatility(df, date(2026, 9, 30), session_count=20) is None


def test_intraday_volatility_currentness_checks_source_and_version():
    from shared_core.rth import intraday_volatility_is_current

    payload = {
        "session_timezone": "America/New_York",
        "session_start": "09:30",
        "session_end": "16:00",
        "lookback_sessions": 20,
        "observations": 20,
        "range_definition": "high_minus_low",
        "range_pct_definition": "(high-low)/rth_open*100",
        "std_method": "sample_ddof_1",
        "median_range_points": 100.0,
        "std_range_points": 20.0,
        "median_range_pct": 0.5,
        "std_range_pct": 0.1,
        "sessions": [{"date": f"2026-09-{i:02d}"} for i in range(1, 21)],
        "calculator_version": "1.0",
        "data_source_id": "nas100-primary",
    }
    assert intraday_volatility_is_current(
        payload,
        calculator_version="1.0",
        session_count=20,
        timezone_name="America/New_York",
        start_hhmm="09:30",
        end_hhmm="16:00",
        data_source_id="nas100-primary",
    )
    assert not intraday_volatility_is_current(
        payload,
        calculator_version="1.1",
        session_count=20,
        timezone_name="America/New_York",
        start_hhmm="09:30",
        end_hhmm="16:00",
        data_source_id="nas100-primary",
    )


def test_analyzer_n_day_summary_uses_latest_n_enriched_sessions():
    from shared_core.rth import summarize_intraday_volatility_payload

    sessions = [
        {"date": f"2026-09-{i:02d}", "range_pct": float(i)}
        for i in range(1, 21)
    ]
    payload = {"sessions": sessions}
    out = summarize_intraday_volatility_payload(payload, 5)
    assert out is not None
    # latest five are 16,17,18,19,20
    assert out["session_count"] == 5
    assert out["median_range_pct"] == 18.0
    expected_std = pd.Series([16.0, 17.0, 18.0, 19.0, 20.0]).std(ddof=1)
    assert abs(out["std_1x_range_pct"] - expected_std) < 1e-12
    assert abs(out["std_2x_range_pct"] - expected_std * 2.0) < 1e-12
    assert abs(out["std_3x_range_pct"] - expected_std * 3.0) < 1e-12
    assert out["start_date"] == "2026-09-16"
    assert out["end_date"] == "2026-09-20"


def test_analyzer_n_day_summary_rejects_outside_2_to_20():
    from shared_core.rth import summarize_intraday_volatility_payload

    payload = {"sessions": [{"date": str(i), "range_pct": float(i)} for i in range(1, 21)]}
    assert summarize_intraday_volatility_payload(payload, 1) is None
    assert summarize_intraday_volatility_payload(payload, 21) is None
