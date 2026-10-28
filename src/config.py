from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

Point = Tuple[int, int]


@dataclass
class Measurement:
    start_lines: list[Tuple[Point, Point]]
    end_lines: list[Tuple[Point, Point]]
    distance_m: float

    @property
    def start_line(self) -> Tuple[Point, Point]:
        return self.start_lines[0]

    @property
    def end_line(self) -> Tuple[Point, Point]:
        return self.end_lines[0]


@dataclass
class LaneConfig:
    lane_id: str
    name: str
    speed_limit_kmph: float
    polygon: List[Point]
    measurement: Measurement


@dataclass
class AppConfig:
    camera_id: str
    evidence_root: str
    pre_event_seconds: float
    post_event_seconds: float
    min_contour_area: int
    max_match_distance_px: float
    fps_override: float | None
    detector_type: str
    yolo_model_path: str
    yolo_confidence: float
    yolo_iou: float
    yolo_classes: list[int]
    lanes: List[LaneConfig]
    video_path: str | None = None


def _to_point_list(raw_points: list[list[int]]) -> List[Point]:
    if not isinstance(raw_points, list) or len(raw_points) < 3:
        raise ValueError("Each lane polygon must contain at least 3 points.")
    return [(int(p[0]), int(p[1])) for p in raw_points]


def _to_line_tuple(raw_line: list | tuple | None, label: str) -> tuple[Point, Point]:
    if not isinstance(raw_line, (list, tuple)) or len(raw_line) != 2:
        raise ValueError(f"{label} must be a list of exactly 2 points: [[x1, y1], [x2, y2]].")
    p1 = raw_line[0]
    p2 = raw_line[1]
    if not isinstance(p1, (list, tuple)) or len(p1) < 2 or not isinstance(p2, (list, tuple)) or len(p2) < 2:
        raise ValueError(f"{label} contains invalid point coordinates.")
    return (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1]))


def _to_line_collection(
    raw_line: list | tuple | None,
    raw_lines: list | tuple | None,
    label: str,
) -> list[tuple[Point, Point]]:
    lines: list[tuple[Point, Point]] = []

    if isinstance(raw_lines, (list, tuple)) and len(raw_lines) > 0:
        for idx, line in enumerate(raw_lines):
            lines.append(_to_line_tuple(line, f"{label}[{idx}]"))
        return lines

    if raw_line is not None:
        lines.append(_to_line_tuple(raw_line, label))

    return lines


def _select_camera(raw: dict, camera_id: str | None, path: Path) -> dict:
    # A calibration file holds {"cameras": [...]} (one entry per camera) or, in the older format, one camera.
    cameras = raw.get("cameras") or [raw]
    if camera_id is None:
        return cameras[0]
    for camera in cameras:
        if str(camera.get("camera_id", "CAM01")) == camera_id:
            return camera
    available = ", ".join(str(c.get("camera_id", "CAM01")) for c in cameras)
    raise ValueError(f"Camera '{camera_id}' not found in {path}. Available cameras: {available}")


def read_cameras(path: str | Path) -> list[dict]:
    """The raw camera entries of a calibration file."""
    with Path(path).open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return raw.get("cameras") or [raw]


def list_camera_ids(path: str | Path) -> list[str]:
    return [str(c.get("camera_id", "CAM01")) for c in read_cameras(path)]


def load_config(path: str | Path, camera_id: str | None = None) -> AppConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = _select_camera(json.load(f), camera_id, path)

    lanes: list[LaneConfig] = []
    for lane in raw.get("lanes", []):
        measurement = lane.get("measurement", {})
        lane_id = lane.get("lane_id", "lane_1")
        polygon = lane.get("polygon")
        if polygon is None:
            raise ValueError(f"Lane '{lane_id}' is missing the 'polygon' field.")
        start_lines = _to_line_collection(
            measurement.get("start_line"),
            measurement.get("start_lines"),
            f"Lane '{lane_id}' start_line",
        )
        end_lines = _to_line_collection(
            measurement.get("end_line"),
            measurement.get("end_lines"),
            f"Lane '{lane_id}' end_line",
        )
        if not start_lines or not end_lines:
            raise ValueError(
                f"Lane '{lane_id}' must define at least one start line and one end line in measurement."
            )

        lanes.append(
            LaneConfig(
                lane_id=lane_id,
                name=lane.get("name", lane_id),
                speed_limit_kmph=float(lane["speed_limit_kmph"]),
                polygon=_to_point_list(polygon),
                measurement=Measurement(
                    start_lines=start_lines,
                    end_lines=end_lines,
                    distance_m=float(measurement["distance_m"]),
                ),
            )
        )

    return AppConfig(
        camera_id=raw.get("camera_id", "CAM01"),
        evidence_root=raw.get("evidence_root", "evidence"),
        pre_event_seconds=float(raw.get("pre_event_seconds", 6)),
        post_event_seconds=float(raw.get("post_event_seconds", 6)),
        min_contour_area=int(raw.get("min_contour_area", 900)),
        max_match_distance_px=float(raw.get("max_match_distance_px", 75)),
        fps_override=float(raw["fps_override"]) if raw.get("fps_override") else None,
        detector_type=raw.get("detector", {}).get("type", "motion"),
        yolo_model_path=raw.get("detector", {}).get("yolo_model_path", "yolo11n.pt"),
        yolo_confidence=float(raw.get("detector", {}).get("confidence", 0.35)),
        yolo_iou=float(raw.get("detector", {}).get("iou", 0.45)),
        yolo_classes=[int(c) for c in raw.get("detector", {}).get("classes", [2, 3, 5, 7])],
        lanes=lanes,
        video_path=raw.get("video_path") or None,
    )
