from pathlib import Path
import json

from pattern_viewer.filtering import collect_patterns, filter_case_entries, load_case_entry, scan_case_entries


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


def test_pattern_filter_any_all(tmp_path):
    _write_case(tmp_path / "a.json", "2026-09-01", "NAS100", ["開盤TR", "BOP"])
    _write_case(tmp_path / "b.json", "2026-09-02", "NAS100", ["開盤TR"])
    _write_case(tmp_path / "c.json", "2026-09-03", "SP500", ["BOM"])

    entries = scan_case_entries(tmp_path)
    assert len(entries) == 3
    assert collect_patterns(entries) == ["BOM", "BOP", "開盤TR"]

    any_rows = filter_case_entries(entries, ["BOP", "BOM"], "ANY")
    assert {x.path.name for x in any_rows} == {"a.json", "c.json"}

    all_rows = filter_case_entries(entries, ["開盤TR", "BOP"], "ALL")
    assert [x.path.name for x in all_rows] == ["a.json"]


def test_invalid_json_is_ignored(tmp_path):
    (tmp_path / "not_case.json").write_text('{"hello": "world"}', encoding="utf-8")
    _write_case(tmp_path / "valid.json", "2026-09-01", "NAS100", ["X"])
    assert load_case_entry(tmp_path / "not_case.json") is None
    rows = scan_case_entries(tmp_path)
    assert [x.path.name for x in rows] == ["valid.json"]
