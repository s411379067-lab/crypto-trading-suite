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



def align_note_x_to_timeframe(m1_timestamp: float, timeframe_seconds: float) -> float:
    """Map an exact M1 note timestamp to the containing displayed candle start.

    The Y anchor can still use the exact M1 close; only the X coordinate is
    bucketed.  This intentionally uses the same epoch-floor convention as the
    chart aggregation code.
    """
    step = float(timeframe_seconds)
    if step <= 0:
        return float(m1_timestamp)
    return float(np.floor(float(m1_timestamp) / step) * step)

def wrap_note_text(text: str, width: int = 26) -> str:
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""
    lines: list[str] = []
    for paragraph in cleaned.splitlines() or [cleaned]:
        wrapped = textwrap.wrap(paragraph, width=max(8, int(width)), replace_whitespace=False, drop_whitespace=False)
        lines.extend(wrapped or [""])
    return "\n".join(lines)


def layout_note_callouts(items: list[dict], max_lanes: int = 4, gap_norm: float = 0.012) -> list[dict]:
    """Assign bulk note callouts to top/bottom lanes without overlapping labels.

    Input items use normalized plot coordinates: ``x_norm`` / ``y_norm`` in [0, 1]
    plus an estimated ``width_norm``.  Anchors in the upper half prefer bottom
    labels and anchors in the lower half prefer top labels, keeping leader lines
    out of the densest candle region.  Within each side, labels are packed into
    the nearest non-overlapping lane and may shift slightly in X when needed.
    """
    lane_count = max(1, int(max_lanes))
    occupancy = {
        "top": [[] for _ in range(lane_count)],
        "bottom": [[] for _ in range(lane_count)],
    }
    out: list[dict] = []

    def clamp_center(center: float, width: float) -> float:
        margin = 0.015
        half = width * 0.5
        lo = margin + half
        hi = 1.0 - margin - half
        if hi < lo:
            return 0.5
        return min(max(center, lo), hi)

    def interval(center: float, width: float) -> tuple[float, float]:
        half = width * 0.5 + float(gap_norm)
        return center - half, center + half

    def overlaps(existing: list[tuple[float, float]], candidate: tuple[float, float]) -> bool:
        left, right = candidate
        return any(not (right <= a or left >= b) for a, b in existing)

    def overlap_cost(existing: list[tuple[float, float]], candidate: tuple[float, float]) -> float:
        left, right = candidate
        total = 0.0
        for a, b in existing:
            total += max(0.0, min(right, b) - max(left, a))
        return total

    # Pack in time order for stable layouts while panning/zooming.
    ordered = sorted(enumerate(items), key=lambda pair: float(pair[1].get("x_norm", 0.0)))
    placements: dict[int, dict] = {}
    shift_candidates = [0.0, -0.04, 0.04, -0.08, 0.08, -0.12, 0.12, -0.18, 0.18]

    for original_idx, item in ordered:
        x = min(max(float(item.get("x_norm", 0.5)), 0.0), 1.0)
        y = min(max(float(item.get("y_norm", 0.5)), 0.0), 1.0)
        width = min(max(float(item.get("width_norm", 0.16)), 0.08), 0.42)
        preferred = "bottom" if y >= 0.5 else "top"
        sides = (preferred, "top" if preferred == "bottom" else "bottom")

        chosen = None
        for side in sides:
            for shift in shift_candidates:
                center = clamp_center(x + shift, width)
                cand = interval(center, width)
                for lane in range(lane_count):
                    if not overlaps(occupancy[side][lane], cand):
                        chosen = (side, lane, center, cand)
                        break
                if chosen is not None:
                    break
            if chosen is not None:
                break

        if chosen is None:
            # Dense fallback: choose the least-overlapping slot while still
            # preferring the side opposite the anchor's vertical half.
            best = None
            for side_rank, side in enumerate(sides):
                for shift in shift_candidates:
                    center = clamp_center(x + shift, width)
                    cand = interval(center, width)
                    for lane in range(lane_count):
                        cost = overlap_cost(occupancy[side][lane], cand)
                        score = (cost, side_rank, abs(shift), lane)
                        if best is None or score < best[0]:
                            best = (score, side, lane, center, cand)
            assert best is not None
            _, side, lane, center, cand = best
            chosen = (side, lane, center, cand)

        side, lane, center, cand = chosen
        occupancy[side][lane].append(cand)
        placements[original_idx] = {
            "side": side,
            "lane": int(lane),
            "center_x_norm": float(center),
        }

    for idx in range(len(items)):
        out.append(placements[idx])
    return out


def layout_alternating_note_callouts(items: list[dict], edge_margin: float = 0.08) -> list[dict]:
    """Deterministic bulk-note layout used by v1.20.

    Notes are ordered by X/time, then assigned Top / Bottom / Top / Bottom.
    Within each side, labels are distributed from left to right in the same
    chronological order.  This keeps leader lines slanted and avoids same-side
    crossings while leaving the candle area mostly unobstructed.
    """
    if not items:
        return []
    margin = min(max(float(edge_margin), 0.0), 0.45)
    ordered = sorted(enumerate(items), key=lambda pair: float(pair[1].get("x_norm", 0.0)))
    top: list[tuple[int, dict]] = []
    bottom: list[tuple[int, dict]] = []
    for sequence_idx, pair in enumerate(ordered):
        (top if sequence_idx % 2 == 0 else bottom).append(pair)

    placements: dict[int, dict] = {}

    def assign(group: list[tuple[int, dict]], side: str) -> None:
        count = len(group)
        if not count:
            return
        if count == 1:
            original_idx, item = group[0]
            x = min(max(float(item.get("x_norm", 0.5)), margin), 1.0 - margin)
            placements[original_idx] = {"side": side, "center_x_norm": x}
            return
        usable = max(0.0, 1.0 - 2.0 * margin)
        for rank, (original_idx, _item) in enumerate(group):
            x = margin + usable * (rank / (count - 1))
            placements[original_idx] = {"side": side, "center_x_norm": x}

    assign(top, "top")
    assign(bottom, "bottom")
    return [placements[i] for i in range(len(items))]
