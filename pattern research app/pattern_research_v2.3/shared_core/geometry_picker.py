from __future__ import annotations

import math


def point_to_segment_distance(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> float:
    """Euclidean distance from point P to finite segment AB, in caller units."""
    px = float(px); py = float(py)
    ax = float(ax); ay = float(ay)
    bx = float(bx); by = float(by)
    dx = bx - ax
    dy = by - ay
    denom = dx * dx + dy * dy
    if denom <= 1e-18:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / denom
    t = min(1.0, max(0.0, t))
    qx = ax + t * dx
    qy = ay + t * dy
    return math.hypot(px - qx, py - qy)


def point_to_rect_distance(
    px: float,
    py: float,
    left: float,
    top: float,
    right: float,
    bottom: float,
    *,
    interior_is_hit: bool = True,
) -> float:
    """Distance to an axis-aligned rectangle.

    When interior_is_hit is True, points inside return 0 (Text Box behavior).
    When False, distance is measured to the nearest edge (Rectangle behavior).
    Coordinates may use either screen-y direction; bounds are normalized.
    """
    px = float(px); py = float(py)
    x0, x1 = sorted((float(left), float(right)))
    y0, y1 = sorted((float(top), float(bottom)))

    inside_x = x0 <= px <= x1
    inside_y = y0 <= py <= y1
    if interior_is_hit and inside_x and inside_y:
        return 0.0

    if not interior_is_hit:
        return min(
            point_to_segment_distance(px, py, x0, y0, x1, y0),
            point_to_segment_distance(px, py, x1, y0, x1, y1),
            point_to_segment_distance(px, py, x1, y1, x0, y1),
            point_to_segment_distance(px, py, x0, y1, x0, y0),
        )

    # Outside distance to the filled rectangle.
    dx = max(x0 - px, 0.0, px - x1)
    dy = max(y0 - py, 0.0, py - y1)
    return math.hypot(dx, dy)
