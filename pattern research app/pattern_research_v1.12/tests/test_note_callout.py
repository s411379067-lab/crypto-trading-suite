from datetime import datetime, timezone

import pandas as pd

from shared_core.note_callout import align_note_x_to_timeframe, find_m1_close, resolve_note_timestamp, wrap_note_text


def test_resolve_new_note_utc_time():
    note = {"replay_time_utc": "2026-09-03T13:30:00Z", "replay_time": "ignored"}
    expected = datetime(2026, 9, 3, 13, 30, tzinfo=timezone.utc).timestamp()
    assert resolve_note_timestamp(note, "America/New_York") == expected


def test_resolve_legacy_new_york_note():
    note = {"replay_time": "2026-09-03 09:30:00 EDT"}
    expected = datetime(2026, 9, 3, 13, 30, tzinfo=timezone.utc).timestamp()
    assert resolve_note_timestamp(note, "UTC") == expected


def test_find_exact_m1_close_and_reject_m5_source():
    base = datetime(2026, 9, 3, 13, 30, tzinfo=timezone.utc).timestamp()
    m1 = pd.DataFrame({
        "timestamp": [base, base + 60, base + 120],
        "close": [100.0, 101.5, 102.0],
    })
    assert find_m1_close(m1, base + 60) == (base + 60, 101.5)

    m5 = pd.DataFrame({
        "timestamp": [base, base + 300, base + 600],
        "close": [100.0, 105.0, 110.0],
    })
    assert find_m1_close(m5, base + 300) is None


def test_wrap_note_text_preserves_content():
    text = "盤中紀錄測試這是一段比較長的文字用來確認會被換行"
    wrapped = wrap_note_text(text, width=10)
    assert "\n" in wrapped
    assert wrapped.replace("\n", "") == text


def test_note_x_aligns_to_display_candle_bucket():
    # 12:59:00 UTC belongs to the 12:55 M5 candle, while M1 stays at 12:59.
    ts = datetime(2026, 9, 3, 12, 59, tzinfo=timezone.utc).timestamp()
    expected_m5 = datetime(2026, 9, 3, 12, 55, tzinfo=timezone.utc).timestamp()
    assert align_note_x_to_timeframe(ts, 60) == ts
    assert align_note_x_to_timeframe(ts, 300) == expected_m5


def test_note_x_aligns_across_common_timeframes():
    ts = datetime(2026, 9, 3, 12, 59, tzinfo=timezone.utc).timestamp()
    assert align_note_x_to_timeframe(ts, 900) == datetime(2026, 9, 3, 12, 45, tzinfo=timezone.utc).timestamp()
    assert align_note_x_to_timeframe(ts, 1800) == datetime(2026, 9, 3, 12, 30, tzinfo=timezone.utc).timestamp()
    assert align_note_x_to_timeframe(ts, 3600) == datetime(2026, 9, 3, 12, 0, tzinfo=timezone.utc).timestamp()
