from __future__ import annotations

import os
import traceback
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
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


def _below_normal_priority() -> None:
    """Clips are saved while the camera keeps measuring; measuring comes first."""
    if os.name != "nt":
        return
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32")
    kernel32.GetCurrentThread.restype = wintypes.HANDLE
    kernel32.SetThreadPriority.argtypes = [wintypes.HANDLE, ctypes.c_int]
    kernel32.SetThreadPriority(kernel32.GetCurrentThread(), -1)  # THREAD_PRIORITY_BELOW_NORMAL


@dataclass
class ActiveRecording:
    event: SpeedEvent
    # Each clip frame with its capture time and the offender's box in it (None when not tracked there).
    frames: list[tuple[np.ndarray, float, BBox | None]]
    triggered_at: datetime
    complete: bool = False


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
        self.pre_event_seconds = pre_event_seconds
        self.post_event_seconds = post_event_seconds
        # The clip covers pre_event_seconds before and post_event_seconds after the measurement, by capture time.
        # From a live camera only the frames that were processed are kept (about 10 per second, not the camera's
        # 25), so a count of frames would cover too long a time and the clip would play too fast.
        # pre_frames is the most there can be: every frame of a recorded video.
        self.pre_frames = max(1, int(pre_event_seconds * fps) + 1)
        self.buffer: Deque[tuple[int, float, np.ndarray]] = deque(maxlen=self.pre_frames)
        self.active_recordings: list[ActiveRecording] = []
        # Saving a 4-megapixel clip takes seconds; done here, the camera would stop measuring meanwhile.
        self._saver = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"evidence-{camera_id}", initializer=_below_normal_priority
        )
        self._saving: list[Future] = []

    def push_frame(self, frame: np.ndarray, frame_index: int, timestamp_s: float, track_boxes: dict[int, BBox]) -> None:
        # One copy per frame, shared by the buffer and every recording (frames are never modified in place),
        # so memory does not grow with the number of overlapping recordings.
        stored = frame.copy()
        self.buffer.append((frame_index, timestamp_s, stored))
        while timestamp_s - self.buffer[0][1] > self.pre_event_seconds:
            self.buffer.popleft()
        for recording in self.active_recordings:
            if not recording.complete:
                recording.frames.append((stored, timestamp_s, track_boxes.get(recording.event.track_id)))
                recording.complete = timestamp_s - recording.event.timestamp_s >= self.post_event_seconds

    def trigger(self, event: SpeedEvent, boxes_by_frame: dict[int, BBox]) -> None:
        """Start a clip for `event` from the buffered frames; `boxes_by_frame` maps frame index -> offender box."""
        frames = [(frame, ts, boxes_by_frame.get(index)) for index, ts, frame in self.buffer]
        self.active_recordings.append(
            ActiveRecording(
                event=event, frames=frames, triggered_at=datetime.now(), complete=self.post_event_seconds <= 0
            )
        )

    def flush_completed(self) -> list[SpeedEvent]:
        """Hand finished recordings to the saving thread; returns the events whose clips are saved by now."""
        pending: list[ActiveRecording] = []
        for recording in self.active_recordings:
            if recording.complete:
                self._saving.append(self._saver.submit(self._save_event, recording))
            else:
                pending.append(recording)
        self.active_recordings = pending
        done = [future for future in self._saving if future.done()]
        self._saving = [future for future in self._saving if not future.done()]
        return [future.result() for future in done]

    def finalize(self) -> list[SpeedEvent]:
        """Save every clip still being recorded or saved, and wait for all of them."""
        for recording in self.active_recordings:
            self._saving.append(self._saver.submit(self._save_event, recording))
        self.active_recordings = []
        completed = [future.result() for future in self._saving]
        self._saving = []
        self._saver.shutdown(wait=True)
        return completed

    def _save_event(self, recording: ActiveRecording) -> SpeedEvent:
        try:
            recording.event.clip_path = str(self._save(recording))
        except Exception:  # the measurement is still recorded, without a clip
            print(f"Camera {self.camera_id}: evidence clip could not be saved:\n{traceback.format_exc()}", flush=True)
            recording.event.clip_path = ""
        return recording.event

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

        # Played back at the rate the frames were really taken, so the clip shows real time: from a live camera
        # that is the rate they were processed at, not the camera's own frame rate.
        span = frames[-1][1] - frames[0][1]
        clip_fps = (len(frames) - 1) / span if len(frames) > 1 and span > 0 else self.fps
        clip_fps = round(min(max(clip_fps, 1.0), self.fps), 2)
        height, width = frames[0][0].shape[:2]
        writer = _open_clip_writer(clip_path, clip_fps, (width, height))

        for frame, _, bbox in frames:
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
