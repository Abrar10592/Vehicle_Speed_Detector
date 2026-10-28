"""Live view files shared by the processing service and the operator dashboard.

For each camera the service keeps <live_dir>/<camera>.jpg (newest overlay frame) and <camera>.json (status)
up to date. A STOP file in <live_dir> asks the service to finish (pending evidence clips are still saved).
"""

from __future__ import annotations

import json
import os
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

STOP_FILE_NAME = "STOP"


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    try:
        os.replace(tmp, path)
    except PermissionError:
        pass  # Windows: the dashboard is reading the old file right now; the next update replaces it.


def write_status(live_dir: str | Path, camera_id: str, **fields) -> None:
    status = {"camera_id": camera_id, "updated": time.time(), **fields}
    _write_atomic(Path(live_dir) / f"{camera_id}.json", json.dumps(status).encode("utf-8"))


def read_status(live_dir: str | Path, camera_id: str) -> dict | None:
    try:
        return json.loads((Path(live_dir) / f"{camera_id}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def read_frame(live_dir: str | Path, camera_id: str) -> bytes | None:
    try:
        return (Path(live_dir) / f"{camera_id}.jpg").read_bytes()
    except OSError:
        return None


class LiveWriter:
    """Publishes one camera's overlay frames (a few per second, downscaled) and status for the dashboard."""

    def __init__(
        self,
        live_dir: str | Path,
        camera_id: str,
        source: str,
        frame_interval_s: float = 0.2,
        max_width: int = 960,
    ) -> None:
        self.live_dir = Path(live_dir)
        self.live_dir.mkdir(parents=True, exist_ok=True)
        self.camera_id = camera_id
        self.frame_interval_s = frame_interval_s
        self.max_width = max_width
        self.fields: dict = {"source": source, "state": "starting", "frames": 0, "fps": 0.0, "speeders": 0}
        self._frame_times: deque[float] = deque(maxlen=30)
        self._last_frame_write = 0.0
        self._last_status_write = 0.0
        self.status("starting")

    def status(self, state: str, **fields) -> None:
        self.fields.update(state=state, **fields)
        write_status(self.live_dir, self.camera_id, **self.fields)
        self._last_status_write = time.monotonic()

    def frame(self, overlay: np.ndarray, frames: int, speeders: int) -> None:
        now = time.monotonic()
        self._frame_times.append(now)
        if now - self._last_frame_write >= self.frame_interval_s:
            height, width = overlay.shape[:2]
            if width > self.max_width:
                overlay = cv2.resize(overlay, (self.max_width, int(height * self.max_width / width)))
            ok, jpg = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ok:
                _write_atomic(self.live_dir / f"{self.camera_id}.jpg", jpg.tobytes())
            self._last_frame_write = now
        if now - self._last_status_write >= 1.0:
            span = self._frame_times[-1] - self._frame_times[0]
            fps = (len(self._frame_times) - 1) / span if span > 0 else 0.0
            self.status("running", frames=frames, fps=round(fps, 1), speeders=speeders)
