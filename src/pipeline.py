from __future__ import annotations

import argparse
import csv
import threading
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor, wait
from functools import partial
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from .config import AppConfig, LaneConfig, list_camera_ids, load_config
from .detector import MotionVehicleDetector, YoloVehicleDetector
from .evidence import EvidenceRecorder
from .geometry import crossed_line
from .live import STOP_FILE_NAME, LiveWriter, write_status
from .models import SpeedEvent
from .tracker import CentroidTracker
from .video_source import open_source

BBox = tuple[int, int, int, int]

EVENT_CSV_FIELDS = [
    "camera_id",
    "track_id",
    "lane_id",
    "speed_kmph",
    "speed_limit_kmph",
    "timestamp_s",
    "frame_index",
    "clip_path",
    "bbox",
    "track_path",
]

# BGR colors for the overlay
COLOR_TRACKED = (255, 180, 0)
COLOR_MEASURING = (0, 255, 255)
COLOR_OK = (0, 200, 0)
COLOR_SPEEDING = (0, 0, 255)


def _point_in_polygon(point: tuple[int, int], polygon: list[tuple[int, int]]) -> bool:
    arr = np.array(polygon, dtype=np.int32)
    return cv2.pointPolygonTest(arr, point, False) >= 0


def _build_lane_roi_mask(frame_shape: tuple[int, ...], config: AppConfig) -> np.ndarray:
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    for lane in config.lanes:
        poly = np.array(lane.polygon, dtype=np.int32)
        cv2.fillPoly(mask, [poly], 255)
    return mask


def _lane_for_point(point: tuple[int, int], lanes: list[LaneConfig]) -> LaneConfig | None:
    for lane in lanes:
        if _point_in_polygon(point, lane.polygon):
            return lane
    return None


# Cameras processed together share one CSV, so writes to it are serialized.
_CSV_LOCK = threading.Lock()


def _prepare_event_csv(csv_path: Path) -> None:
    """Move aside a CSV written with an older column layout, so new rows don't land under the wrong headers."""
    with _CSV_LOCK:
        if not csv_path.exists():
            return
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            header = next(csv.reader(f), None)
        if header and header != EVENT_CSV_FIELDS:
            csv_path.replace(csv_path.with_name(f"{csv_path.stem}_old{csv_path.suffix}"))


def _append_event_csv(csv_path: Path, event: SpeedEvent) -> None:
    with _CSV_LOCK, csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EVENT_CSV_FIELDS)
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(asdict(event))


def _draw_overlay(
    frame: np.ndarray,
    config: AppConfig,
    track_boxes: dict[int, BBox],
    track_centroids: dict[int, tuple[int, int]],
    measured_by_track: dict[int, SpeedEvent],
    measuring_ids: set[int],
    recent_events: list[SpeedEvent],
    debug_state_by_track: dict[int, str] | None = None,
    debug_overlay: bool = False,
) -> np.ndarray:
    out = frame.copy()
    for lane in config.lanes:
        poly = np.array(lane.polygon, dtype=np.int32)
        cv2.polylines(out, [poly], isClosed=True, color=(0, 255, 255), thickness=2)

        for start_line in lane.measurement.start_lines:
            cv2.line(out, start_line[0], start_line[1], (0, 255, 0), 2)
        for end_line in lane.measurement.end_lines:
            cv2.line(out, end_line[0], end_line[1], (0, 0, 255), 2)

        label_point = tuple(poly[0])
        cv2.putText(
            out,
            f"{lane.name} limit={lane.speed_limit_kmph:.0f}",
            (label_point[0], max(20, label_point[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    for track_id, (x1, y1, x2, y2) in track_boxes.items():
        event = measured_by_track.get(track_id)
        if event is not None:
            color = COLOR_SPEEDING if event.speed_kmph > event.speed_limit_kmph else COLOR_OK
        elif track_id in measuring_ids:
            color = COLOR_MEASURING
        else:
            color = COLOR_TRACKED

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        if track_id in track_centroids:
            cv2.circle(out, track_centroids[track_id], 3, (0, 255, 255), -1)
        cv2.putText(
            out,
            f"ID {track_id}",
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Speed is only known once the vehicle has crossed both lines; until then show that it is being timed.
        if event is not None:
            label = f"{event.speed_kmph:.1f} km/h"
        elif track_id in measuring_ids:
            label = "measuring..."
        else:
            label = None
        if label is not None:
            cv2.putText(
                out,
                label,
                (x1, max(40, y1 + 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

        if debug_overlay:
            debug_text = ""
            if debug_state_by_track and track_id in debug_state_by_track:
                debug_text = debug_state_by_track[track_id]
            else:
                debug_text = "lane=- in=0 start=0 end=0"

            cv2.putText(
                out,
                debug_text,
                (x1, min(out.shape[0] - 8, y2 + 16)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

    # Panel with the latest measurements, so every speed stays readable after its vehicle leaves the frame.
    if recent_events:
        line_h = 22
        panel_w, panel_h = 300, 30 + line_h * len(recent_events)
        panel = out[10 : 10 + panel_h, 10 : 10 + panel_w]
        out[10 : 10 + panel_h, 10 : 10 + panel_w] = (panel * 0.35).astype(np.uint8)
        cv2.putText(out, "Recent speeds", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        for i, event in enumerate(recent_events):
            color = COLOR_SPEEDING if event.speed_kmph > event.speed_limit_kmph else COLOR_OK
            cv2.putText(
                out,
                f"ID {event.track_id}  {event.lane_id}  {event.speed_kmph:.1f} km/h",
                (20, 32 + line_h * (i + 1)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

    return out


def run(
    video_path: str | None,
    config_path: str,
    output_overlay_path: str | None = None,
    debug_overlay: bool = False,
    max_frames: int | None = None,
    camera_id: str | None = None,
    stop_event: threading.Event | None = None,
    live_dir: str | None = None,
    realtime: bool = False,
) -> None:
    """Process one camera's video file or live stream.

    live_dir: publish overlay frames and status there for the operator dashboard; a STOP file there ends the run.
    realtime: process a video file no faster than it was recorded (a stand-in for a live camera).
    """
    config = load_config(config_path, camera_id)
    if not config.lanes:
        raise RuntimeError(f"Camera '{config.camera_id}' has no calibrated lanes in {config_path}.")
    video_path = video_path or config.video_path
    if not video_path:
        raise RuntimeError(f"No video given: pass --video or set a video for camera '{config.camera_id}' in the dashboard.")
    print(f"Camera {config.camera_id}: {video_path}")
    live = LiveWriter(live_dir, config.camera_id, video_path) if live_dir else None
    stop_file = Path(live_dir) / STOP_FILE_NAME if live_dir else None
    source = open_source(video_path, config.fps_override, realtime)
    fps = source.fps

    def stop_requested() -> bool:
        return bool(stop_event and stop_event.is_set()) or bool(stop_file and stop_file.exists())

    def next_frame() -> tuple[np.ndarray, float] | None:
        # A live stream that drops out is waited for until it is back or the run is stopped.
        while not stop_requested():
            item = source.read()
            if item is not None or not source.live:
                return item
            if live is not None:
                live.status("reconnecting")
        return None

    if config.detector_type.lower() == "yolo":
        detector = YoloVehicleDetector(
            model_path=config.yolo_model_path,
            confidence=config.yolo_confidence,
            iou=config.yolo_iou,
            classes=config.yolo_classes,
        )
        print(
            f"Camera {config.camera_id}: using YOLO detector, model={config.yolo_model_path}, conf={config.yolo_confidence}, iou={config.yolo_iou}"
        )
    else:
        detector = MotionVehicleDetector(min_contour_area=config.min_contour_area)
        print(f"Camera {config.camera_id}: using motion detector (background subtraction).")
    tracker = CentroidTracker(max_match_distance_px=config.max_match_distance_px)
    evidence = EvidenceRecorder(
        evidence_root=config.evidence_root,
        camera_id=config.camera_id,
        fps=fps,
        pre_event_seconds=config.pre_event_seconds,
        post_event_seconds=config.post_event_seconds,
    )

    first = next_frame()
    if first is None:
        source.release()
        if stop_requested():
            return
        raise RuntimeError("Video has no readable frames.")
    first_frame, first_timestamp_s = first

    frame_h, frame_w = first_frame.shape[:2]
    lane_mask = _build_lane_roi_mask(first_frame.shape, config)

    overlay_writer = None
    if output_overlay_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        overlay_writer = cv2.VideoWriter(output_overlay_path, fourcc, fps, (frame_w, frame_h))

    events_csv_path = Path(config.evidence_root) / "speed_events.csv"
    events_csv_path.parent.mkdir(parents=True, exist_ok=True)
    _prepare_event_csv(events_csv_path)

    lane_track_state: dict[tuple[int, str], dict[str, float | bool | int]] = {}
    prev_centroids: dict[int, tuple[int, int]] = {}
    measured_by_track: dict[int, SpeedEvent] = {}
    # (frame_index, bbox) per track; long enough to cover the evidence pre-event window.
    history_len = max(120, evidence.pre_frames + 1)
    track_history: dict[int, deque[tuple[int, BBox]]] = {}
    recent_events: deque[SpeedEvent] = deque(maxlen=8)

    frame_idx = 0

    stats = {"speeders": 0}

    def process_frame(frame: np.ndarray, frame_index: int, timestamp_s: float) -> None:

        detections = detector.detect(frame, roi_mask=lane_mask)
        tracks = tracker.update(detections, ts_s=timestamp_s)
        track_boxes: dict[int, BBox] = {}
        # Boxes of vehicles actually detected in this frame (the tracker also returns recently lost tracks at
        # their last position); evidence clips only mark the offender where it was really seen.
        observed_boxes: dict[int, BBox] = {}
        track_centroids: dict[int, tuple[int, int]] = {}
        debug_state_by_track: dict[int, str] = {}
        measuring_ids: set[int] = set()
        speeding: list[tuple[SpeedEvent, dict[int, BBox]]] = []

        for track in tracks:
            x1, y1, x2, y2 = track.bbox
            bbox = (int(x1), int(y1), int(x2), int(y2))
            track_boxes[track.track_id] = bbox
            track_centroids[track.track_id] = track.centroid

            history = track_history.setdefault(track.track_id, deque(maxlen=history_len))
            if track.last_seen_ts == timestamp_s:
                observed_boxes[track.track_id] = bbox
                history.append((frame_index, bbox))

            lane = _lane_for_point(track.centroid, config.lanes)
            if lane is None:
                debug_state_by_track[track.track_id] = "lane=- in=0 start=0 end=0"
                prev_centroids[track.track_id] = track.centroid
                continue

            key = (track.track_id, lane.lane_id)
            if key not in lane_track_state:
                lane_track_state[key] = {
                    "start_crossed": False,
                    "start_ts": 0.0,
                    "start_frame": 0,
                    "done": False,
                }

            if track.track_id not in prev_centroids:
                prev_centroids[track.track_id] = track.centroid
                continue

            previous = prev_centroids[track.track_id]
            current = track.centroid
            state = lane_track_state[key]

            crossed_any_start = any(
                crossed_line(previous, current, start_line) for start_line in lane.measurement.start_lines
            )
            crossed_any_end = any(
                crossed_line(previous, current, end_line) for end_line in lane.measurement.end_lines
            )

            if not state["start_crossed"] and crossed_any_start:
                state["start_crossed"] = True
                state["start_ts"] = timestamp_s
                state["start_frame"] = frame_index

            if (
                state["start_crossed"]
                and not state["done"]
                and crossed_any_end
            ):
                elapsed_s = max(0.001, timestamp_s - float(state["start_ts"]))
                speed_kmph = (lane.measurement.distance_m / elapsed_s) * 3.6

                event = SpeedEvent(
                    camera_id=config.camera_id,
                    track_id=track.track_id,
                    lane_id=lane.lane_id,
                    speed_kmph=float(speed_kmph),
                    speed_limit_kmph=float(lane.speed_limit_kmph),
                    timestamp_s=timestamp_s,
                    frame_index=frame_index,
                    bbox=bbox,
                    track_path=[b for _, b in history],
                )
                measured_by_track[track.track_id] = event
                recent_events.appendleft(event)

                # Speeding events are written to the CSV once their evidence clip is saved (with its path).
                if speed_kmph > lane.speed_limit_kmph:
                    speeding.append((event, dict(history)))
                    stats["speeders"] += 1
                else:
                    _append_event_csv(events_csv_path, event)

                state["done"] = True

            if state["start_crossed"] and not state["done"]:
                measuring_ids.add(track.track_id)

            debug_state_by_track[track.track_id] = (
                f"lane={lane.lane_id} in=1 start={int(bool(state['start_crossed']))} end={int(bool(state['done']))}"
            )

            prev_centroids[track.track_id] = track.centroid

        evidence.push_frame(frame, frame_index, observed_boxes)
        for event, boxes_by_frame in speeding:
            evidence.trigger(event, boxes_by_frame)

        for completed_event in evidence.flush_completed():
            _append_event_csv(events_csv_path, completed_event)

        if overlay_writer is not None or live is not None:
            overlay = _draw_overlay(
                frame,
                config,
                track_boxes,
                track_centroids,
                measured_by_track,
                measuring_ids,
                list(recent_events),
                debug_state_by_track,
                debug_overlay,
            )
            if overlay_writer is not None:
                overlay_writer.write(overlay)
            if live is not None:
                live.frame(overlay, frame_index + 1, stats["speeders"])

        # The tracker never reuses an ID, so state for tracks it has dropped can be released.
        for track_id in [t for t in track_history if t not in track_boxes]:
            del track_history[track_id]
            prev_centroids.pop(track_id, None)
            measured_by_track.pop(track_id, None)
        for key in [k for k in lane_track_state if k[0] not in track_boxes]:
            del lane_track_state[key]

    process_frame(first_frame, frame_idx, first_timestamp_s)
    frame_idx += 1

    while max_frames is None or frame_idx < max_frames:
        item = next_frame()
        if item is None:
            break
        process_frame(item[0], frame_idx, item[1])
        frame_idx += 1

    for completed_event in evidence.finalize():
        _append_event_csv(events_csv_path, completed_event)

    source.release()
    if overlay_writer is not None:
        overlay_writer.release()
    if live is not None:
        live.status("stopped" if stop_requested() else "finished", frames=frame_idx, speeders=stats["speeders"])
    print(f"Camera {config.camera_id}: finished after {frame_idx} frames.")


def _report_failure(live_dir: str, camera_id: str, future: Future) -> None:
    if future.exception() is not None:
        write_status(live_dir, camera_id, state="error", error=str(future.exception()))


def run_all_cameras(
    config_path: str,
    output_overlay_path: str | None = None,
    debug_overlay: bool = False,
    max_frames: int | None = None,
    live_dir: str | None = None,
    realtime: bool = False,
) -> None:
    """Process every camera in the config at the same time (one thread each), each with its own video."""
    camera_ids = list_camera_ids(config_path)
    stop_event = threading.Event()

    def overlay_path_for(camera_id: str) -> str | None:
        if not output_overlay_path:
            return None
        path = Path(output_overlay_path)
        return str(path.with_name(f"{path.stem}_{camera_id}{path.suffix}"))

    with ThreadPoolExecutor(max_workers=len(camera_ids)) as pool:
        futures = {}
        for camera_id in camera_ids:
            future = pool.submit(
                run,
                None,
                config_path,
                overlay_path_for(camera_id),
                debug_overlay,
                max_frames,
                camera_id,
                stop_event,
                live_dir,
                realtime,
            )
            if live_dir:
                # Shown on the operator dashboard straight away, while the other cameras keep running.
                future.add_done_callback(partial(_report_failure, live_dir, camera_id))
            futures[future] = camera_id
        try:
            # Wait in short steps so Ctrl+C is noticed; the cameras then stop and save what they have.
            while wait(futures, timeout=0.5).not_done:
                pass
        except KeyboardInterrupt:
            print("Stopping all cameras...")
            stop_event.set()

    errors = {camera_id: future.exception() for future, camera_id in futures.items() if future.exception()}
    for camera_id, error in errors.items():
        print(f"Camera {camera_id} failed: {error}")
    if errors:
        raise SystemExit(1)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vehicle speed detector for recorded videos")
    parser.add_argument("--video", default=None, help="Path to recorded video (default: the camera's video in the config)")
    parser.add_argument("--config", required=True, help="Path to calibration JSON")
    parser.add_argument(
        "--camera",
        default=None,
        help="Camera name in the config (default: the first camera), or 'all' to process every camera at the same time",
    )
    parser.add_argument("--overlay-out", default=None, help="Optional output video with overlays")
    parser.add_argument(
        "--debug-overlay",
        action="store_true",
        help="Show per-vehicle debug state (lane membership, start/end crossing flags).",
    )
    parser.add_argument(
        "--live-dir",
        default=None,
        help="Publish live overlay frames and status here for the operator dashboard (a STOP file there stops the run).",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Process video files no faster than they were recorded, like a live camera.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after this many frames (useful for quick tests on long videos).",
    )
    return parser.parse_args()


def main() -> None:
    args = _args()
    if args.camera == "all":
        if args.video:
            raise SystemExit("--video cannot be used with --camera all: each camera uses its own video from the config.")
        run_all_cameras(
            config_path=args.config,
            output_overlay_path=args.overlay_out,
            debug_overlay=args.debug_overlay,
            max_frames=args.max_frames,
            live_dir=args.live_dir,
            realtime=args.realtime,
        )
        return
    run(
        video_path=args.video,
        config_path=args.config,
        output_overlay_path=args.overlay_out,
        debug_overlay=args.debug_overlay,
        max_frames=args.max_frames,
        camera_id=args.camera,
        live_dir=args.live_dir,
        realtime=args.realtime,
    )


if __name__ == "__main__":
    main()
