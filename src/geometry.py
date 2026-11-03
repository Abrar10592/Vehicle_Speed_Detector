from __future__ import annotations

from typing import Tuple

Point = Tuple[int, int]
Line = Tuple[Point, Point]


def road_point(bbox: tuple[int, int, int, int]) -> Point:
    """Where a vehicle meets the road in the image: the bottom center of its box.

    Lanes and lines are drawn on the road surface. The center of the box floats above the road, by a large
    and changing amount on a low camera, so it would reach the lines much later than the vehicle or never.
    """
    x1, _, x2, y2 = bbox
    return (x1 + x2) // 2, y2


def _orientation(a: Point, b: Point, c: Point) -> int:
    value = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
    if value == 0:
        return 0
    return 1 if value > 0 else 2


def _on_segment(a: Point, b: Point, c: Point) -> bool:
    return (
        min(a[0], c[0]) <= b[0] <= max(a[0], c[0])
        and min(a[1], c[1]) <= b[1] <= max(a[1], c[1])
    )


def segments_intersect(p1: Point, q1: Point, p2: Point, q2: Point) -> bool:
    o1 = _orientation(p1, q1, p2)
    o2 = _orientation(p1, q1, q2)
    o3 = _orientation(p2, q2, p1)
    o4 = _orientation(p2, q2, q1)

    if o1 != o2 and o3 != o4:
        return True

    if o1 == 0 and _on_segment(p1, p2, q1):
        return True
    if o2 == 0 and _on_segment(p1, q2, q1):
        return True
    if o3 == 0 and _on_segment(p2, p1, q2):
        return True
    if o4 == 0 and _on_segment(p2, q1, q2):
        return True

    return False


def crossed_line(previous: Point, current: Point, line: Line) -> bool:
    return segments_intersect(previous, current, line[0], line[1])
