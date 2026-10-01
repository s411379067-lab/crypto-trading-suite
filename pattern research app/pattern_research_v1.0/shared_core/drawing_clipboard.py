from __future__ import annotations

from copy import deepcopy
from typing import Any

from .models import new_id


def clone_drawing_with_offset(drawing: dict[str, Any], dx: float, dy: float) -> dict[str, Any]:
    """Deep-copy a drawing, assign a new id, and offset only its geometry.

    Style, text content, Fibo levels, and template-derived settings are preserved.
    The function is UI-library agnostic and safe to unit test without Qt.
    """
    clone = deepcopy(drawing)
    clone["id"] = new_id("drawing")
    dtype = str(clone.get("type", ""))
    dx = float(dx)
    dy = float(dy)

    if dtype == "horizontal_line":
        clone["price"] = float(clone.get("price", 0.0)) + dy

    elif dtype in {"trend_line", "rectangle"}:
        points = clone.get("points", [])
        for point in points:
            point["time"] = float(point.get("time", 0.0)) + dx
            point["price"] = float(point.get("price", 0.0)) + dy

    elif dtype == "fibonacci":
        for key in ("start", "end"):
            point = clone.get(key)
            if isinstance(point, dict):
                point["time"] = float(point.get("time", 0.0)) + dx
                point["price"] = float(point.get("price", 0.0)) + dy

    elif dtype == "text":
        clone["time"] = float(clone.get("time", 0.0)) + dx
        clone["price"] = float(clone.get("price", 0.0)) + dy

    return clone
