from pathlib import Path
import json

from pattern_viewer.filtering import (
    collect_patterns,
    filter_case_entries,
    load_case_entry,
    save_case_patterns,
    scan_case_entries,
)


def _write_case(path: Path, date: str, symbol: str, patterns: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "case": {"id": path.stem, "symbol": symbol, "research_date": date},
        "market_data": {"source_path": "raw.txt"},
        "time_range": {
            "data_start": f"{date}T08:00:00-04:00",
            "replay_start": f"{date}T09:30:00-04:00",
            "default_end": f"{date}T16:00:00-04:00",
        },
        "patterns": [{"text": p} for p in patterns],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_pattern_filter_any_all_and_exclusion_modes(tmp_path):
    _write_case(tmp_path / "a.json", "2026-09-01", "NAS100", ["開盤TR", "BOP"])
    _write_case(tmp_path / "b.json", "2026-09-02", "NAS100", ["開盤TR"])
    _write_case(tmp_path / "c.json", "2026-09-03", "SP500", ["BOM"])
    _write_case(tmp_path / "d.json", "2026-09-04", "SP500", [])

    entries = scan_case_entries(tmp_path)
    assert len(entries) == 4
    assert collect_patterns(entries) == ["BOM", "BOP", "開盤TR"]

    any_rows = filter_case_entries(entries, ["BOP", "BOM"], "ANY")
    assert {x.path.name for x in any_rows} == {"a.json", "c.json"}

    all_rows = filter_case_entries(entries, ["開盤TR", "BOP"], "ALL")
    assert [x.path.name for x in all_rows] == ["a.json"]

    exclude_any_rows = filter_case_entries(entries, ["開盤TR", "BOM"], "EXCLUDE_ANY")
    assert [x.path.name for x in exclude_any_rows] == ["d.json"]

    exclude_all_rows = filter_case_entries(entries, ["開盤TR", "BOP"], "EXCLUDE_ALL")
    assert {x.path.name for x in exclude_all_rows} == {"b.json", "c.json", "d.json"}

    with_patterns_only = filter_case_entries(entries, [], has_patterns_only=True)
    assert {x.path.name for x in with_patterns_only} == {"a.json", "b.json", "c.json"}


def test_invalid_json_is_ignored(tmp_path):
    (tmp_path / "not_case.json").write_text('{"hello": "world"}', encoding="utf-8")
    _write_case(tmp_path / "valid.json", "2026-09-01", "NAS100", ["X"])
    assert load_case_entry(tmp_path / "not_case.json") is None
    rows = scan_case_entries(tmp_path)
    assert [x.path.name for x in rows] == ["valid.json"]


def test_save_case_patterns_changes_only_patterns_and_updated_at(tmp_path):
    path = tmp_path / "case.json"
    payload = {
        "schema_version": "9.9",
        "case": {"id": "case-x", "symbol": "NAS100", "research_date": "2026-09-01"},
        "market_data": {"source_path": "raw.txt", "custom": 123},
        "time_range": {
            "data_start": "2026-09-01T08:00:00-04:00",
            "replay_start": "2026-09-01T09:30:00-04:00",
            "default_end": "2026-09-01T16:00:00-04:00",
        },
        "patterns": [{"id": "old", "text": "OLD"}],
        "drawings": [{"id": "keep-me", "type": "horizontal_line"}],
        "unknown_future_field": {"keep": True},
        "metadata": {"created_at": "created", "updated_at": "old-time", "other": "keep"},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    new_patterns = [{"id": "p1", "text": "開盤TR", "created_at": "a", "updated_at": "b"}]
    save_case_patterns(path, new_patterns)
    out = json.loads(path.read_text(encoding="utf-8"))

    assert out["patterns"] == new_patterns
    assert out["drawings"] == payload["drawings"]
    assert out["unknown_future_field"] == payload["unknown_future_field"]
    assert out["market_data"] == payload["market_data"]
    assert out["metadata"]["created_at"] == "created"
    assert out["metadata"]["other"] == "keep"
    assert out["metadata"]["updated_at"] != "old-time"
