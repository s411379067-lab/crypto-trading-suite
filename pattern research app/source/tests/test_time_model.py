from datetime import date, time

from shared_core.time_model import build_case_time_range
from shared_core.models import ResearchCase


def test_default_end_crosses_midnight():
    tr = build_case_time_range(
        date(2026, 9, 3), time(8, 0), time(9, 30), time(4, 30), "America/New_York"
    )
    assert tr["data_start"].startswith("2026-09-03T08:00:00")
    assert tr["replay_start"].startswith("2026-09-03T09:30:00")
    assert tr["default_end"].startswith("2026-09-04T04:30:00")


def test_replay_start_can_cross_midnight_too():
    tr = build_case_time_range(
        date(2026, 9, 3), time(20, 0), time(9, 30), time(15, 0), "America/New_York"
    )
    assert tr["data_start"].startswith("2026-09-03T20:00:00")
    assert tr["replay_start"].startswith("2026-09-04T09:30:00")
    assert tr["default_end"].startswith("2026-09-04T15:00:00")


def test_legacy_case_gets_stable_time_context():
    data = {
        "schema_version": "0.1",
        "case": {"symbol": "NAS100", "research_date": "2026-09-03"},
        "market_data": {"source_path": "x.txt"},
        "time_range": {
            "data_start": "2026-09-03T08:00:00-04:00",
            "replay_start": "2026-09-03T09:30:00-04:00",
            "default_end": "2026-09-04T04:30:00-04:00",
        },
        "display": {"timezone": "America/New_York"},
    }
    case = ResearchCase.from_dict(data)
    assert case.time_context["case_timezone"] == "America/New_York"
    assert case.time_context["canonical_timezone"] == "UTC"
    case.display["timezone"] = "Asia/Taipei"
    assert case.time_context["case_timezone"] == "America/New_York"
