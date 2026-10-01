from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

from .models import ResearchCase


class CaseRepository:
    def load(self, path: str | Path) -> ResearchCase:
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return ResearchCase.from_dict(data)

    def save(self, path: str | Path, case: ResearchCase) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        case.touch()
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(case.to_dict(), f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    def list_cases(self, root: str | Path) -> Iterable[Path]:
        root = Path(root)
        if not root.exists():
            return []
        return sorted(root.rglob("*.json"))
