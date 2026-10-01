from __future__ import annotations

from typing import Any


def build_order_overlay(orders: list[dict[str, Any]], current_ts: float) -> tuple[list[dict], list[dict]]:
    """Build visible fill markers and completed Open-to-Close chart segments.

    The result is derived from persisted fill records only.  It never mutates the
    Case and respects the current replay timestamp, so unrevealed fills cannot
    appear on the chart.  A partial close produces one segment for the closed
    quantity; a reversing fill closes the old position before opening the new one.
    """
    visible: list[dict] = []
    for order in orders or []:
        if order.get("status") != "filled" or order.get("fill_ts") is None or order.get("fill_price") is None:
            continue
        try:
            fill_ts = float(order["fill_ts"])
            if fill_ts > float(current_ts):
                continue
            visible.append({
                "id": order.get("id"),
                "timestamp": fill_ts,
                "price": float(order["fill_price"]),
                "side": str(order.get("side", "long")),
                "action": order.get("computed_action", ""),
                "qty": float(order.get("qty") or 0.0),
                "created_ts": float(order.get("created_ts") or 0.0),
            })
        except (TypeError, ValueError):
            continue

    visible.sort(key=lambda item: (item["timestamp"], item["created_ts"], str(item.get("id", ""))))
    segments: list[dict] = []
    position: dict | None = None
    for fill in visible:
        qty = float(fill["qty"])
        if qty <= 0:
            continue
        side = fill["side"]
        price = float(fill["price"])
        if position is None:
            position = {"side": side, "entry_price": price, "entry_ts": fill["timestamp"], "qty": qty}
            continue
        if position["side"] == side:
            new_qty = float(position["qty"]) + qty
            position["entry_price"] = (float(position["entry_price"]) * float(position["qty"]) + price * qty) / new_qty
            position["qty"] = new_qty
            continue

        close_qty = min(float(position["qty"]), qty)
        side_mult = 1.0 if position["side"] == "long" else -1.0
        pnl = (price - float(position["entry_price"])) * side_mult * close_qty
        if abs(pnl) > 1e-12:
            segments.append({
                "entry_timestamp": float(position["entry_ts"]),
                "entry_price": float(position["entry_price"]),
                "exit_timestamp": float(fill["timestamp"]),
                "exit_price": price,
                "pnl": pnl,
                "qty": close_qty,
            })
        position["qty"] = float(position["qty"]) - close_qty
        remainder = qty - close_qty
        if position["qty"] <= 1e-12:
            position = None
        if remainder > 1e-12:
            position = {"side": side, "entry_price": price, "entry_ts": fill["timestamp"], "qty": remainder}

    return visible, segments
