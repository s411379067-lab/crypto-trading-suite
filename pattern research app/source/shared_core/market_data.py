from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd


TIME_CANDIDATES = [
    "dt_utc",
    "timestamp",
    "time_utc8",
    "time_utc3",
    "datetime",
    "time",
    "date",
]


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", str(name).strip().lower()).strip("_")


def normalize_market_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        raise ValueError("Market data is empty")

    df = raw_df.copy()
    df.columns = [_normalize_name(c) for c in df.columns]

    time_col = next((c for c in TIME_CANDIDATES if c in df.columns), None)
    if time_col is None:
        raise ValueError(f"Cannot find timestamp column. columns={list(df.columns)}")

    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Market data missing OHLC columns: {missing}")

    dt = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    valid = dt.notna()
    if not valid.any():
        raise ValueError(f"Timestamp column {time_col} cannot be parsed")

    out = pd.DataFrame({
        "dt_utc": dt[valid],
        "open": pd.to_numeric(df.loc[valid, "open"], errors="coerce"),
        "high": pd.to_numeric(df.loc[valid, "high"], errors="coerce"),
        "low": pd.to_numeric(df.loc[valid, "low"], errors="coerce"),
        "close": pd.to_numeric(df.loc[valid, "close"], errors="coerce"),
    })

    for src, dst in [
        ("tick_volume", "tick_volume"),
        ("real_volume", "volume"),
        ("volume", "volume"),
        ("spread", "spread"),
    ]:
        if src in df.columns and dst not in out.columns:
            out[dst] = pd.to_numeric(df.loc[valid, src], errors="coerce")

    out = out.dropna(subset=["open", "high", "low", "close"]).copy()
    out["timestamp"] = out["dt_utc"].astype("int64") / 1_000_000_000.0
    out = out.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    return out


class MarketDataService:
    def __init__(self):
        self._cache: dict[Path, pd.DataFrame] = {}

    def _read_raw(self, path: Path) -> pd.DataFrame:
        suffix = path.suffix.lower()
        if suffix in (".parquet", ".pq"):
            return pd.read_parquet(path)
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix in (".txt", ".tsv"):
            try:
                return pd.read_csv(path, sep=None, engine="python")
            except Exception:
                return pd.read_csv(path, sep="\t")
        raise ValueError(f"Unsupported market data format: {suffix}")

    def load_path(self, path: str | Path) -> pd.DataFrame:
        path = Path(path).expanduser().resolve()
        if path not in self._cache:
            if not path.exists():
                raise FileNotFoundError(path)
            self._cache[path] = normalize_market_dataframe(self._read_raw(path))
        return self._cache[path].copy(deep=False)

    def load_for_case(self, case_path: str | Path, market_data: dict) -> pd.DataFrame:
        source = Path(str(market_data["source_path"])).expanduser()
        if not source.is_absolute():
            source = Path(case_path).resolve().parent / source
        return self.load_path(source)
