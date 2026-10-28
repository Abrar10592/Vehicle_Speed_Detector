from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Deque

import cv2
import numpy as np

from .models import SpeedEvent

BBox = tuple[int, int, int, int]


def _open_clip_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    """Open an H.264 writer, so clips play in the dashboard (browsers can't play OpenCV's default mp4v).

    Windows' own encoder is tried first because OpenCV's FFmpeg build often has no working H.264 encoder;
    mp4v is the last resort.
    """
    attempts = [(cv2.CAP_MSMF, "avc1")] if os.name == "nt" else []
    attempts += [(cv2.CAP_FFMPEG, "avc1"), (cv2.CAP_FFMPEG, "mp4v")]
    for api, codec in attempts:
        writer = cv2.VideoWriter(str(path), api, cv2.VideoWriter_fourcc(*codec), fps, size)
        if writer.isOpened():
            return writer
    raise RuntimeError(f"Cannot write video file {path}")


@dataclass
class ActiveRecording:
    event: SpeedEvent
    # Each clip frame with the offender's box in that frame (None when the vehicle was not tracked there).
    frames: list[tuple[np.ndarray, BBox | None]]
    remaining_post_frames: int
    triggered_at: datetime


class EvidenceRecorder:
    def __init__(
        self,
        evidence_root: str,
        camera_id: str,
        fps: float,
        pre_event_seconds: float,
        post_event_seconds: float,
    ) -> None:
        self.evidence_root = Path(evidence_root)
        self.evidence_root.mkdir(parents=True, exist_ok=True)
        self.camera_id = camera_id
        self.fps = fps
        self.pre_frames = max(1, int(pre_event_seconds * fps))
        self.post_frames = max(1, int(post_event_seconds * fps))
        self.buffer: Deque[tuple[int, np.ndarray]] = deque(maxlen=self.pre_frames)
        self.active_recordings: list[ActiveRecording] = []

    def push_frame(self, frame: np.ndarray, frame_index: int, track_boxes: dict[int, BBox]) -> None:
        # One copy per frame, shared by the buffer and every recording (frames are never modified in place),
        # so memory does not grow with the number of overlapping recordings.
        stored = frame.copy()
        self.buffer.append((frame_index, stored))
        for recording in self.active_recordings:
            if recording.remaining_post_frames > 0:
                recording.frames.append((stored, track_boxes.get(recording.event.track_id)))
                recording.remaining_post_frames -= 1

    def trigger(self, event: SpeedEvent, boxes_by_frame: dict[int, BBox]) -> None:
        """Start a clip for `event` from the buffered frames; `boxes_by_frame` maps frame index -> offender box."""
        frames = [(frame, boxes_by_frame.get(index)) for index, frame in self.buffer]
        self.active_recordings.append(
            ActiveRecording(
                event=event, frames=frames, remaining_post_frames=self.post_frames, triggered_at=datetime.now()
            )
        )

    def flush_completed(self) -> list[SpeedEvent]:
        completed: list[SpeedEvent] = []
        pending: list[ActiveRecording] = []

        for recording in self.active_recordings:
            if recording.remaining_post_frames <= 0:
                recording.event.clip_path = str(self._save(recording))
                completed.append(recording.event)
            else:
                pending.append(recording)

        self.active_recordings = pending
        return completed

    def finalize(self) -> list[SpeedEvent]:
        completed: list[SpeedEvent] = []
        for recording in self.active_recordings:
            recording.event.clip_path = str(self._save(recording))
            completed.append(recording.event)
        self.active_recordings = []
        return completed

    def _save(self, recording: ActiveRecording) -> Path:
        """Write one short clip in which only the offending vehicle is marked.

        Saved as evidence/<date>/<camera>/<time>_<lane>_<speed>kmh.mp4, e.g. 19-14-34_lane_2_115.7kmh.mp4.
        """
        event, frames, when = recording.event, recording.frames, recording.triggered_at
        if not frames:
            raise ValueError("No frames found for evidence clip.")

        day_dir = self.evidence_root / when.strftime("%Y-%m-%d") / self.camera_id
        day_dir.mkdir(parents=True, exist_ok=True)

        base_name = f"{when:%H-%M-%S}_{event.lane_id}_{event.speed_kmph:.1f}kmh"
        clip_path = day_dir / f"{base_name}.mp4"
        suffix = 2
        while clip_path.exists():  # two offenders in the same second, lane and speed
            clip_path = day_dir / f"{base_name}_{suffix}.mp4"
            suffix += 1

        height, width = frames[0][0].shape[:2]
        writer = _open_clip_writer(clip_path, self.fps, (width, height))

        for frame, bbox in frames:
            out = frame.copy()
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(
                    out,
                    f"OFFENDER: {event.speed_kmph:.1f} km/h",
                    (max(10, x1), max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
            writer.write(out)
        writer.release()
        return clip_path
