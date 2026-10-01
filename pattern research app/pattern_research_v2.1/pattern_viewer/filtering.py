from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json


@dataclass(frozen=True)
class ViewerCaseEntry:
    path: Path
    symbol: str
    research_date: str
    patterns: tuple[str, ...]

    @property
    def display_name(self) -> str:
        tag_text = " / ".join(self.patterns) if self.patterns else "無 Pattern"
        return f"{self.research_date}   {self.symbol}   |   {tag_text}"


def _normalized_pattern_text(value) -> str:
    if isinstance(value, dict):
        value = value.get("text", "")
    return str(value or "").strip()


def load_case_entry(path: str | Path) -> ViewerCaseEntry | None:
    """Read only the lightweight metadata needed by Pattern Viewer.

    Invalid/non-Case JSON files are ignored instead of breaking a recursive folder scan.
    """
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    case = data.get("case")
    market_data = data.get("market_data")
    time_range = data.get("time_range")
    if not isinstance(case, dict) or not isinstance(market_data, dict) or not isinstance(time_range, dict):
        return None
    if not market_data.get("source_path"):
        return None
    if not all(time_range.get(k) for k in ("data_start", "replay_start", "default_end")):
        return None

    seen = set()
    patterns: list[str] = []
    for raw in data.get("patterns", []) or []:
        text = _normalized_pattern_text(raw)
        if text and text not in seen:
            patterns.append(text)
            seen.add(text)

    return ViewerCaseEntry(
        path=path.resolve(),
        symbol=str(case.get("symbol", "")),
        research_date=str(case.get("research_date", "")),
        patterns=tuple(patterns),
    )


def scan_case_entries(root: str | Path) -> list[ViewerCaseEntry]:
    root = Path(root).expanduser()
    if not root.exists():
        return []
    out = []
    for path in sorted(root.rglob("*.json")):
        entry = load_case_entry(path)
        if entry is not None:
            out.append(entry)
    out.sort(key=lambda x: (x.research_date, x.symbol, str(x.path)), reverse=True)
    return out


def collect_patterns(entries: list[ViewerCaseEntry]) -> list[str]:
    return sorted({p for entry in entries for p in entry.patterns}, key=lambda s: s.casefold())


def filter_case_entries(
    entries: list[ViewerCaseEntry],
    selected_patterns: list[str] | tuple[str, ...] | set[str],
    mode: str = "ANY",
) -> list[ViewerCaseEntry]:
    selected = {str(x).strip() for x in selected_patterns if str(x).strip()}
    if not selected:
        return list(entries)

    mode = str(mode or "ANY").upper()
    out = []
    for entry in entries:
        tags = set(entry.patterns)
        matched = selected.issubset(tags) if mode == "ALL" else bool(selected & tags)
        if matched:
            out.append(entry)
    return out
