from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
import uuid


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


@dataclass
class ResearchCase:
    schema_version: str
    case: dict[str, Any]
    market_data: dict[str, Any]
    time_range: dict[str, Any]
    time_context: dict[str, Any]
    display: dict[str, Any]
    replay: dict[str, Any]
    calendar: dict[str, Any] = field(default_factory=dict)
    reference_levels: dict[str, Any] = field(default_factory=dict)
    patterns: list[dict[str, Any]] = field(default_factory=list)
    intraday_notes: list[dict[str, Any]] = field(default_factory=list)
    drawings: list[dict[str, Any]] = field(default_factory=list)
    orders: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResearchCase":
        required = ["case", "market_data", "time_range"]
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"Case JSON missing required fields: {missing}")

        # Backward compatibility: older cases had no explicit time_context.
        # We infer it once from their saved display timezone, then persist it on
        # the next save. Later display-timezone changes do not alter this value.
        legacy_display = dict(data.get("display", {}))
        inferred_case_tz = legacy_display.get("timezone", "UTC")
        time_context = dict(data.get("time_context", {}))
        time_context.setdefault("case_timezone", inferred_case_tz)
        time_context.setdefault("canonical_timezone", "UTC")

        case = cls(
            schema_version=str(data.get("schema_version", "0.1")),
            case=dict(data["case"]),
            market_data=dict(data["market_data"]),
            time_range=dict(data["time_range"]),
            time_context=time_context,
            display=legacy_display,
            replay=dict(data.get("replay", {})),
            calendar=dict(data.get("calendar", {})),
            reference_levels=dict(data.get("reference_levels", {})),
            patterns=list(data.get("patterns", [])),
            intraday_notes=list(data.get("intraday_notes", [])),
            drawings=list(data.get("drawings", [])),
            orders=list(data.get("orders", [])),
            metadata=dict(data.get("metadata", {})),
        )
        case._apply_defaults()
        case.validate()
        return case

    def _apply_defaults(self) -> None:
        self.case.setdefault("id", new_id("case"))
        self.case.setdefault("symbol", "UNKNOWN")
        self.case.setdefault("research_date", "")

        self.market_data.setdefault("source_id", "primary")
        self.market_data.setdefault("source_type", "txt")
        self.market_data.setdefault("resolution", "M1")

        self.time_context.setdefault("case_timezone", self.display.get("timezone", "UTC"))
        self.time_context.setdefault("canonical_timezone", "UTC")

        self.display.setdefault("view_timeframe", "M5")
        self.display.setdefault("timezone", self.time_context.get("case_timezone", "UTC"))
        self.display.setdefault("x_tick_interval", "15m")

        replay_start = self.time_range.get("replay_start")
        self.replay.setdefault("step_minutes", 1)
        self.replay.setdefault("current_time", replay_start)

        self.metadata.setdefault("created_at", utc_now_iso())
        self.metadata.setdefault("updated_at", self.metadata["created_at"])

    def validate(self) -> None:
        for key in ("data_start", "replay_start", "default_end"):
            if not self.time_range.get(key):
                raise ValueError(f"time_range.{key} is required")
        if not self.market_data.get("source_path"):
            raise ValueError("market_data.source_path is required in v0.1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "case": self.case,
            "market_data": self.market_data,
            "time_range": self.time_range,
            "time_context": self.time_context,
            "display": self.display,
            "replay": self.replay,
            "calendar": self.calendar,
            "reference_levels": self.reference_levels,
            "patterns": self.patterns,
            "intraday_notes": self.intraday_notes,
            "drawings": self.drawings,
            "orders": self.orders,
            "metadata": self.metadata,
        }

    def touch(self) -> None:
        self.metadata["updated_at"] = utc_now_iso()

    def add_pattern(self, text: str) -> dict[str, Any]:
        item = {
            "id": new_id("pattern"),
            "text": text.strip(),
            "created_at": utc_now_iso(),
            "updated_at": utc_now_iso(),
        }
        self.patterns.append(item)
        self.touch()
        return item

    def add_note(self, replay_time: str, text: str) -> dict[str, Any]:
        item = {
            "id": new_id("note"),
            "replay_time": replay_time,
            "text": text.strip(),
            "created_at": utc_now_iso(),
            "updated_at": utc_now_iso(),
        }
        self.intraday_notes.append(item)
        self.touch()
        return item
