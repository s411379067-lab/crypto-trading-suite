from datetime import datetime, timezone

import pandas as pd

from shared_core.note_callout import find_m1_close, resolve_note_timestamp, wrap_note_text


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
