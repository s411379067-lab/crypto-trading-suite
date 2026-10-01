from __future__ import annotations

from pathlib import Path


def default_case_root() -> Path:
    """Return the local Case-data folder beside the fixed source folder."""
    return Path(__file__).resolve().parents[2] / "cases"
