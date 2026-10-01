from __future__ import annotations

from datetime import datetime
import re
import textwrap
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


_ABBREVIATION_ZONES = {
    "EDT": "America/New_York",
    "EST": "America/New_York",
    "CST": "Asia/Taipei",
    "BST": "Europe/London",
    "GMT": "UTC",
    "UTC": "UTC",
}


def resolve_note_timestamp(note: dict, fallback_timezone: str = "UTC") -> float | None:
    """Resolve a note replay time to canonical UTC epoch seconds.

    New notes may contain ``replay_time_utc``.  Older notes only have the
    human-readable replay_time string, so this function keeps a compatibility
    parser for the display-zone abbreviations used by the application.
    """
    canonical = str(note.get("replay_time_utc", "") or "").strip()
    if canonical:
        try:
            return float(pd.Timestamp(canonical).timestamp())
        except Exception:
            pass

    value = str(note.get("replay_time", "") or "").strip()
    if not value:
        return None

    # Offset-aware / Zulu ISO strings are already unambiguous.
    if value.endswith("Z") or re.search(r"[+-]\d{2}:?\d{2}$", value):
        try:
            return float(pd.Timestamp(value).timestamp())
        except Exception:
            return None

    zone_name = fallback_timezone or "UTC"
    base = value
    match = re.match(r"^(.*?)(?:\s+([A-Za-z]{2,5}))$", value)
    if match:
        candidate_base, abbreviation = match.group(1).strip(), match.group(2).upper()
        if abbreviation in _ABBREVIATION_ZONES:
            base = candidate_base
            zone_name = _ABBREVIATION_ZONES[abbreviation]

    try:
        naive = pd.Timestamp(base).to_pydatetime()
        if naive.tzinfo is not None:
            return float(naive.timestamp())
        aware = naive.replace(tzinfo=ZoneInfo(zone_name))
        return float(aware.timestamp())
    except Exception:
        return None


def find_m1_close(raw_df: pd.DataFrame, target_ts: float, tolerance_seconds: float = 2.0) -> tuple[float, float] | None:
    """Return (bar timestamp, close) for the M1 bar at target_ts.

    The raw source must actually look like one-minute data.  We intentionally do
    not fall back to a nearby five-minute candle because the callout definition is
    explicitly tied to the one-minute close.
    """
    if raw_df is None or raw_df.empty or "timestamp" not in raw_df.columns or "close" not in raw_df.columns:
        return None

    ts = raw_df["timestamp"].to_numpy(dtype=float)
    if ts.size == 0:
        return None
    order = np.argsort(ts)
    ts_sorted = ts[order]

    diffs = np.diff(ts_sorted)
    diffs = diffs[(diffs > 0) & (diffs <= 600)]
    if diffs.size:
        median_resolution = float(np.median(diffs))
        if median_resolution > 90.0:
            return None

    idx = int(np.searchsorted(ts_sorted, float(target_ts)))
    candidates = []
    if idx < ts_sorted.size:
        candidates.append(idx)
    if idx > 0:
        candidates.append(idx - 1)
    if not candidates:
        return None

    best = min(candidates, key=lambda i: abs(float(ts_sorted[i]) - float(target_ts)))
    actual_ts = float(ts_sorted[best])
    if abs(actual_ts - float(target_ts)) > float(tolerance_seconds):
        return None

    original_index = int(order[best])
    try:
        close = float(raw_df.iloc[original_index]["close"])
    except Exception:
        return None
    return actual_ts, close


def wrap_note_text(text: str, width: int = 26) -> str:
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""
    lines: list[str] = []
    for paragraph in cleaned.splitlines() or [cleaned]:
        wrapped = textwrap.wrap(paragraph, width=max(8, int(width)), replace_whitespace=False, drop_whitespace=False)
        lines.extend(wrapped or [""])
    return "\n".join(lines)
