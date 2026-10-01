from __future__ import annotations
import math
from typing import Any
import pandas as pd

def summarize_current_replay_range(raw_df: pd.DataFrame, replay_start_ts: float, current_ts: float) -> dict[str, Any] | None:
    if raw_df is None or raw_df.empty:
        return None
    required={"timestamp","open","high","low"}
    if not required.issubset(raw_df.columns):
        return None
    try:
        start=float(replay_start_ts); end=float(current_ts)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(start) and math.isfinite(end)) or end < start:
        return None
    ts=pd.to_numeric(raw_df["timestamp"], errors="coerce")
    frame=raw_df.loc[(ts>=start-1e-9)&(ts<=end+1e-9), ["timestamp","open","high","low"]].copy()
    if frame.empty: return None
    frame=frame.sort_values("timestamp", kind="stable")
    for c in ("open","high","low"):
        frame[c]=pd.to_numeric(frame[c], errors="coerce")
    frame=frame.dropna(subset=["open","high","low"])
    if frame.empty: return None
    base_open=float(frame.iloc[0]["open"]); high=float(frame["high"].max()); low=float(frame["low"].min())
    if base_open == 0 or not all(math.isfinite(v) for v in (base_open, high, low)): return None
    rp=high-low
    return {"bar_count":int(len(frame)),"base_open":base_open,"high":high,"low":low,"range_points":float(rp),"range_pct":float(rp/base_open*100.0),"start_ts":float(frame.iloc[0]["timestamp"]),"end_ts":float(frame.iloc[-1]["timestamp"])}
