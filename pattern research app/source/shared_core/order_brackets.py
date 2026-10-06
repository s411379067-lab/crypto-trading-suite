from __future__ import annotations

import uuid


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
