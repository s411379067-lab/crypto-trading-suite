from __future__ import annotations

import math
from typing import Mapping, Any


def candle_change_pct(open_price: float, close_price: float) -> float | None:
    """Return candle close-vs-open percentage change, or None when undefined."""
    try:
        open_value = float(open_price)
        close_value = float(close_price)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(open_value) or not math.isfinite(close_value) or abs(open_value) < 1e-15:
        return None
    return (close_value - open_value) / open_value * 100.0


def format_ohlc_text(row: Mapping[str, Any] | None, *, price_decimals: int = 2, pct_decimals: int = 3) -> str:
    """Format one displayed candle for the Analyzer/Viewer OHLC information row."""
    if row is None:
        return "開=--  高=--  低=--  收=--  漲跌=--"
    try:
        open_value = float(row["open"])
        high_value = float(row["high"])
        low_value = float(row["low"])
        close_value = float(row["close"])
    except (KeyError, TypeError, ValueError):
        return "開=--  高=--  低=--  收=--  漲跌=--"

    change = candle_change_pct(open_value, close_value)
    change_text = "--" if change is None else f"{change:+.{pct_decimals}f}%"
    text = (
        f"開={open_value:.{price_decimals}f}   高={high_value:.{price_decimals}f}   "
        f"低={low_value:.{price_decimals}f}   收={close_value:.{price_decimals}f}   漲跌={change_text}"
    )
    try:
        partial = bool(row.get("is_partial", False))
    except AttributeError:
        partial = False
    if partial:
        text += "   [forming]"
    return text
