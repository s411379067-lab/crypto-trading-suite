from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any


RESEARCH_FIELDS = ("drawings", "patterns", "intraday_notes")


def research_snapshot(case) -> dict[str, Any]:
    """Return the undoable research state only.

    Replay position, display settings, RTH visibility, orders and other UI/session
    state are deliberately excluded from the global research undo history.
    """
    return {name: deepcopy(getattr(case, name, [])) for name in RESEARCH_FIELDS}


def apply_research_snapshot(case, snapshot: dict[str, Any]) -> None:
    for name in RESEARCH_FIELDS:
        setattr(case, name, deepcopy(snapshot.get(name, [])))


@dataclass
class UndoEntry:
    label: str
    before: dict[str, Any]
    after: dict[str, Any]


class ResearchUndoManager:
    def __init__(self, max_depth: int = 200):
        self.max_depth = max(1, int(max_depth))
        self.undo_stack: list[UndoEntry] = []
        self.redo_stack: list[UndoEntry] = []
        self._baseline: dict[str, Any] | None = None

    def reset(self, case) -> None:
        self.undo_stack.clear()
        self.redo_stack.clear()
        self._baseline = research_snapshot(case) if case is not None else None

    def record(self, case, label: str) -> bool:
        if case is None:
            return False
        current = research_snapshot(case)
        if self._baseline is None:
            self._baseline = current
            return False
        if current == self._baseline:
            return False
        self.undo_stack.append(UndoEntry(str(label or "Edit"), deepcopy(self._baseline), deepcopy(current)))
        if len(self.undo_stack) > self.max_depth:
            self.undo_stack = self.undo_stack[-self.max_depth:]
        self.redo_stack.clear()
        self._baseline = current
        return True

    def can_undo(self) -> bool:
        return bool(self.undo_stack)

    def can_redo(self) -> bool:
        return bool(self.redo_stack)

    def undo(self, case) -> str | None:
        if case is None or not self.undo_stack:
            return None
        entry = self.undo_stack.pop()
        apply_research_snapshot(case, entry.before)
        self.redo_stack.append(entry)
        self._baseline = research_snapshot(case)
        return entry.label

    def redo(self, case) -> str | None:
        if case is None or not self.redo_stack:
            return None
        entry = self.redo_stack.pop()
        apply_research_snapshot(case, entry.after)
        self.undo_stack.append(entry)
        self._baseline = research_snapshot(case)
        return entry.label
