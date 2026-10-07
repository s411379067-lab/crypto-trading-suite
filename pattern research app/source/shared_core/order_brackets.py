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


def set_bracket_group_qty(bracket: dict, group_id: str, qty: float) -> dict:
    """Change one group's shared SL/TP lots without exceeding Entry lots."""
    updated = deepcopy(bracket)
    qty = float(qty)
    if qty <= 0:
        raise ValueError("Group quantity must be positive")
    for group in updated.get("groups", []):
        if str(group.get("id")) != str(group_id):
            continue
        other_qty = allocated_bracket_qty(updated) - float(group.get("qty", 0.0))
        if other_qty + qty > float(updated["qty"]) + 1e-9:
            raise ValueError("Group quantity exceeds available Entry lots")
        group["qty"] = qty
        return updated
    raise ValueError("Bracket group was not found")


def remove_bracket_group(bracket: dict, group_id: str) -> dict:
    """Return a bracket without the selected SL/TP group."""
    updated = deepcopy(bracket)
    before = len(updated.get("groups", []))
    updated["groups"] = [g for g in updated.get("groups", []) if str(g.get("id")) != str(group_id)]
    if len(updated["groups"]) == before:
        raise ValueError("Bracket group was not found")
    return updated


def add_bracket_group(bracket: dict, qty: float | None = None) -> dict:
    """Add an SL/TP pair using available lots and the initial symmetric distance."""
    updated = deepcopy(bracket)
    available = available_bracket_qty(updated)
    qty = available if qty is None else float(qty)
    if qty <= 0:
        raise ValueError("No unallocated lots are available for another SL/TP group")
    if qty > available + 1e-9:
        raise ValueError("Group quantity exceeds available Entry lots")
    entry = float(updated["entry_price"])
    distance = max(abs(entry) * 0.01, 0.0001)
    if str(updated.get("side")) == "short":
        stop, target = entry + distance, entry - distance
    else:
        stop, target = entry - distance, entry + distance
    updated.setdefault("groups", []).append({
        "id": f"bracket-group-{uuid.uuid4().hex[:12]}",
        "qty": qty,
        "stop_price": stop,
        "target_price": target,
    })
    return updated


def estimated_bracket_pnl(bracket: dict, price: float, qty: float) -> float:
    """Estimate linear PnL at a bracket price, in quote currency dollars."""
    direction = 1.0 if str(bracket.get("side")) == "long" else -1.0
    return (float(price) - float(bracket["entry_price"])) * direction * float(qty)


def entry_order_type_for_price(side: str, entry_price: float, current_price: float) -> str:
    """Select limit or stop-market according to Entry's position vs current price."""
    side = "short" if str(side) == "short" else "long"
    entry_price = float(entry_price)
    current_price = float(current_price)
    if current_price > entry_price:
        return "stop market" if side == "short" else "limit"
    return "limit" if side == "short" else "stop market"


def pending_bracket_order_specs(bracket: dict) -> list[dict]:
    """Build the Entry plus linked SL/TP pending-order instructions."""
    side = "short" if str(bracket.get("side")) == "short" else "long"
    entry_type = str(bracket.get("order_type") or "limit")
    entry_price = float(bracket["entry_price"])
    entry_qty = float(bracket["qty"])
    if entry_qty <= 0:
        raise ValueError("Bracket quantity must be positive")
    close_side = "long" if side == "short" else "short"
    specs = [{
        "role": "entry", "group_id": None, "side": side,
        "order_type": entry_type, "price": entry_price, "qty": entry_qty,
    }]
    for group in bracket.get("groups", []):
        qty = float(group["qty"])
        if qty <= 0:
            raise ValueError("Group quantity must be positive")
        group_id = str(group["id"])
        specs.extend((
            {"role": "stop", "group_id": group_id, "side": close_side,
             "order_type": "stop market", "price": float(group["stop_price"]), "qty": qty},
            {"role": "target", "group_id": group_id, "side": close_side,
             "order_type": "limit", "price": float(group["target_price"]), "qty": qty},
        ))
    return specs
