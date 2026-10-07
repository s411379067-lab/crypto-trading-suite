from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import Any, Iterable


@dataclass(frozen=True)
class ViewerMetrics:
    trades: int
    wins: int
    trading_days: int
    total_pnl: float
    profit_factor: float | None
    win_rate: float | None
    pnl_per_day: float | None
    pnl_per_trade: float | None


def completed_trade_pnls(orders: Iterable[dict[str, Any]] | None) -> list[float]:
    """Return one realized PnL per fill event that closes existing quantity.

    Same-side fills are combined into a weighted-average entry. Opposite-side
    fills close the current position FIFO-style, matching the chart overlay's
    realized PnL calculation. Breakeven closes are retained for win-rate counts.
    """
    fills = []
    for order in orders or []:
        if not isinstance(order, dict) or order.get("status") != "filled":
            continue
        try:
            timestamp = float(order["fill_ts"])
            price = float(order["fill_price"])
            qty = float(order.get("qty") or 0.0)
            if qty <= 0:
                continue
            fills.append((timestamp, float(order.get("created_ts") or 0.0), str(order.get("id", "")),
                          str(order.get("side", "long")), price, qty))
        except (KeyError, TypeError, ValueError):
            continue
    fills.sort(key=lambda fill: (fill[0], fill[1], fill[2]))

    realized: list[float] = []
    position: dict[str, float | str] | None = None
    for _timestamp, _created, _order_id, side, price, qty in fills:
        if position is None:
            position = {"side": side, "entry": price, "qty": qty}
            continue
        if position["side"] == side:
            old_qty = float(position["qty"])
            new_qty = old_qty + qty
            position["entry"] = (float(position["entry"]) * old_qty + price * qty) / new_qty
            position["qty"] = new_qty
            continue

        close_qty = min(float(position["qty"]), qty)
        direction = 1.0 if position["side"] == "long" else -1.0
        realized.append((price - float(position["entry"])) * direction * close_qty)
        remaining_position = float(position["qty"]) - close_qty
        remaining_fill = qty - close_qty
        if remaining_position <= 1e-12:
            position = None
        else:
            position["qty"] = remaining_position
        if remaining_fill > 1e-12:
            position = {"side": side, "entry": price, "qty": remaining_fill}
    return realized


def calculate_viewer_metrics(cases: Iterable[tuple[str, Iterable[float]]]) -> ViewerMetrics:
    """Aggregate realized closes by filtered Case; research_date defines a day."""
    daily_pnl: dict[str, float] = {}
    pnls: list[float] = []
    for research_date, case_pnls in cases:
        case_pnls = [float(pnl) for pnl in case_pnls]
        if not case_pnls:
            continue
        date_key = str(research_date or "").strip()
        if not date_key:
            date_key = "(unknown date)"
        daily_pnl[date_key] = daily_pnl.get(date_key, 0.0) + sum(case_pnls)
        pnls.extend(case_pnls)

    trades = len(pnls)
    wins = sum(pnl > 1e-12 for pnl in pnls)
    gross_profit = sum(pnl for pnl in pnls if pnl > 1e-12)
    gross_loss = -sum(pnl for pnl in pnls if pnl < -1e-12)
    total_pnl = sum(pnls)
    trading_days = len(daily_pnl)
    if gross_loss > 1e-12:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 1e-12:
        profit_factor = inf
    else:
        profit_factor = None

    return ViewerMetrics(
        trades=trades,
        wins=wins,
        trading_days=trading_days,
        total_pnl=total_pnl,
        profit_factor=profit_factor,
        win_rate=(wins / trades) if trades else None,
        pnl_per_day=(total_pnl / trading_days) if trading_days else None,
        pnl_per_trade=(total_pnl / trades) if trades else None,
    )
