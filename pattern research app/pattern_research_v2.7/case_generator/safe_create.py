from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def find_existing_case_file(folder: Path, day_str: str) -> Path | None:
    """Return an existing Case file for *day_str* without modifying anything.

    Enricher may rename weekend files from ``YYYY-MM-DD.json`` to
    ``YYYY-MM-DD(W).json``.  Generator therefore treats either filename as an
    already-existing Case for that research date.
    """
    for candidate in (
        folder / f"{day_str}.json",
        folder / f"{day_str}(W).json",
    ):
        if candidate.exists():
            return candidate
    return None


def create_case_json_exclusive(path: Path, case: dict[str, Any]) -> bool:
    """Create a Case JSON only when *path* does not already exist.

    ``x`` mode maps to an exclusive create at the OS level, so even if another
    process creates the file after the caller's existence check, this function
    will refuse to overwrite it.

    Returns True when a new file is created, False when the destination already
    exists.  Existing file contents are never opened for writing.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as fh:
            json.dump(case, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
    except FileExistsError:
        return False
    return True
