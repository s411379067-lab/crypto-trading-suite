from __future__ import annotations

import numpy as np
import pandas as pd


TF_SECONDS = {
    "M1": 60,
    "M5": 300,
    "M15": 900,
    "M30": 1800,
    "H1": 3600,
    "H4": 14400,
}


def timeframe_seconds(tf: str) -> int:
    key = tf.strip().upper()
    if key not in TF_SECONDS:
        raise ValueError(f"Unsupported timeframe: {tf}")
    return TF_SECONDS[key]


def aggregate_visible_bars(raw: pd.DataFrame, timeframe: str, current_ts: float, start_ts: float) -> pd.DataFrame:
    if raw.empty:
        return raw.copy()

    visible = raw[(raw["timestamp"] >= start_ts) & (raw["timestamp"] <= current_ts)].copy()
    if visible.empty:
        return visible

    tf_sec = timeframe_seconds(timeframe)
    if tf_sec == 60:
        return visible.reset_index(drop=True)

    visible["bucket"] = np.floor(visible["timestamp"].to_numpy(dtype=float) / tf_sec) * tf_sec
    agg_map = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "timestamp": "first",
    }
    if "tick_volume" in visible.columns:
        agg_map["tick_volume"] = "sum"
    if "volume" in visible.columns:
        agg_map["volume"] = "sum"
    if "spread" in visible.columns:
        agg_map["spread"] = "last"

    grouped = visible.groupby("bucket", sort=True).agg(agg_map).reset_index(drop=False)
    grouped["timestamp"] = grouped["bucket"].astype(float)
    grouped["dt_utc"] = pd.to_datetime(grouped["timestamp"], unit="s", utc=True)
    grouped["is_partial"] = (grouped["bucket"] + tf_sec - 60) > current_ts
    return grouped.drop(columns=["bucket"]).reset_index(drop=True)
