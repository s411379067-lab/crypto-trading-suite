from __future__ import annotations

import uuid
from copy import deepcopy


def create_pending_bracket(side: str, entry_price: float, qty: float, risk_fraction: float = 0.01) -> dict:
    """Create the initial, session-only Entry + one SL/TP group layout.

    The initial stop and target are symmetric around Entry.  Later UI stages may
    move either leg independently; this helper deliberately owns only creation.
    """
    side = "short" if str(side) == "short" else "long"
    entry_price = float(entry_price)
    qty = float(qty)
    if qty <= 0:
        raise ValueError("Bracket quantity must be positive")
    distance = max(abs(entry_price) * float(risk_fraction), 0.0001)
    if side == "short":
        stop_price, target_price = entry_price + distance, entry_price - distance
    else:
        stop_price, target_price = entry_price - distance, entry_price + distance
    return {
        "id": f"bracket-{uuid.uuid4().hex[:12]}",
        "state": "pending",
        "side": side,
        "entry_price": entry_price,
        "qty": qty,
        "groups": [{
            "id": f"bracket-group-{uuid.uuid4().hex[:12]}",
            "qty": qty,
            "stop_price": stop_price,
            "target_price": target_price,
        }],
    }


def move_bracket_entry(bracket: dict, entry_price: float) -> dict:
    """Move Entry and retain every group's SL/TP distance from Entry."""
    updated = deepcopy(bracket)
    old_entry = float(updated["entry_price"])
    entry_price = float(entry_price)
    delta = entry_price - old_entry
    updated["entry_price"] = entry_price
    for group in updated.get("groups", []):
        group["stop_price"] = float(group["stop_price"]) + delta
        group["target_price"] = float(group["target_price"]) + delta
    return updated


def move_bracket_leg(bracket: dict, group_id: str, leg: str, price: float) -> dict:
    """Move one SL or TP leg without changing the other leg in its group."""
    key = {"stop": "stop_price", "target": "target_price"}.get(str(leg))
    if key is None:
        raise ValueError("Bracket leg must be stop or target")
    updated = deepcopy(bracket)
    for group in updated.get("groups", []):
        if str(group.get("id")) == str(group_id):
            group[key] = float(price)
            return updated
    raise ValueError("Bracket group was not found")


def allocated_bracket_qty(bracket: dict) -> float:
    """Return the total lots reserved by SL/TP groups."""
    return sum(float(group.get("qty", 0.0)) for group in bracket.get("groups", []))


def available_bracket_qty(bracket: dict) -> float:
    """Return lots that can still receive another SL/TP group."""
    return max(0.0, float(bracket["qty"]) - allocated_bracket_qty(bracket))


def set_bracket_qty(bracket: dict, qty: float) -> dict:
    """Change Entry lots without allowing existing SL/TP allocations to overflow."""
    updated = deepcopy(bracket)
    qty = float(qty)
    if qty <= 0:
        raise ValueError("Bracket quantity must be positive")
    if qty + 1e-9 < allocated_bracket_qty(updated):
        raise ValueError("Bracket quantity cannot be less than allocated SL/TP lots")
    updated["qty"] = qty
    return updated
