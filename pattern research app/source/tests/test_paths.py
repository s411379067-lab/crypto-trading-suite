from pathlib import Path

from shared_core.paths import default_case_root


def test_default_case_root_is_beside_source_folder():
    expected = Path(__file__).resolve().parents[2] / "cases"
    assert default_case_root() == expected
