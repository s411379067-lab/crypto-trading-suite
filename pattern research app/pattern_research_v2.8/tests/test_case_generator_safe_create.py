from __future__ import annotations

import json
from pathlib import Path

from case_generator.safe_create import create_case_json_exclusive, find_existing_case_file


def test_create_case_json_exclusive_creates_new_file(tmp_path: Path):
    path = tmp_path / "2026-09-22.json"
    case = {"patterns": [{"text": "new"}], "drawings": []}

    assert create_case_json_exclusive(path, case) is True
    assert json.loads(path.read_text(encoding="utf-8")) == case


def test_create_case_json_exclusive_never_overwrites_existing_research(tmp_path: Path):
    path = tmp_path / "2026-09-20.json"
    original = {
        "patterns": [{"text": "Opening TR"}],
        "intraday_notes": [{"text": "keep me"}],
        "drawings": [{"id": "drawing-1", "type": "line"}],
        "orders": [{"id": "fill-1"}],
    }
    original_text = json.dumps(original, ensure_ascii=False, indent=2)
    path.write_text(original_text, encoding="utf-8")

    replacement = {"patterns": [], "intraday_notes": [], "drawings": [], "orders": []}
    assert create_case_json_exclusive(path, replacement) is False

    # Byte-for-byte preservation is the contract for an existing Case.
    assert path.read_text(encoding="utf-8") == original_text


def test_existing_weekend_w_file_blocks_duplicate_plain_case(tmp_path: Path):
    weekend = tmp_path / "2026-09-26(W).json"
    weekend.write_text('{"drawings":[{"id":"keep"}]}', encoding="utf-8")

    assert find_existing_case_file(tmp_path, "2026-09-26") == weekend


def test_plain_existing_case_is_detected_first(tmp_path: Path):
    plain = tmp_path / "2026-09-25.json"
    plain.write_text("{}", encoding="utf-8")

    assert find_existing_case_file(tmp_path, "2026-09-25") == plain


def test_missing_case_returns_none(tmp_path: Path):
    assert find_existing_case_file(tmp_path, "2026-09-24") is None
