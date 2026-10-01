from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


def iso_to_ts(value: str) -> float:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return float(ts.tz_convert("UTC").timestamp())


def ts_to_iso(value: float) -> str:
    return pd.Timestamp(value, unit="s", tz="UTC").isoformat()


@dataclass
class ReplayEngine:
    data_start_ts: float
    replay_start_ts: float
    default_end_ts: float
    current_ts: float
    step_seconds: int
    max_data_ts: float

    @classmethod
    def from_case(cls, case, max_data_ts: float) -> "ReplayEngine":
        data_start = iso_to_ts(case.time_range["data_start"])
        replay_start = iso_to_ts(case.time_range["replay_start"])
        default_end = iso_to_ts(case.time_range["default_end"])
        current = iso_to_ts(case.replay.get("current_time") or case.time_range["replay_start"])
        current = max(replay_start, min(current, max_data_ts))
        return cls(
            data_start_ts=data_start,
            replay_start_ts=replay_start,
            default_end_ts=default_end,
            current_ts=current,
            step_seconds=max(60, int(case.replay.get("step_minutes", 1)) * 60),
            max_data_ts=max_data_ts,
        )

    def forward(self) -> float:
        self.current_ts = min(self.current_ts + self.step_seconds, self.max_data_ts)
        return self.current_ts

    def backward(self) -> float:
        self.current_ts = max(self.replay_start_ts, self.current_ts - self.step_seconds)
        return self.current_ts

    def reset(self) -> float:
        self.current_ts = self.replay_start_ts
        return self.current_ts
