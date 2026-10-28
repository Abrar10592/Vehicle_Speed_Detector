from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

Point = Tuple[int, int]
BBox = Tuple[int, int, int, int]


@dataclass
class Detection:
    bbox: BBox
    score: float
    class_name: str = "vehicle"


@dataclass
class Track:
    track_id: int
    bbox: BBox
    centroid: Point
    last_seen_ts: float


@dataclass
class SpeedEvent:
    camera_id: str
    track_id: int
    lane_id: str
    speed_kmph: float
    speed_limit_kmph: float
    timestamp_s: float
    frame_index: int
    clip_path: str | None = None
    bbox: tuple[int, int, int, int] | None = None
    track_path: list[tuple[int, int, int, int]] | None = None
