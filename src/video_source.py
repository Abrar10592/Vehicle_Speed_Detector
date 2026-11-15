"""Frame sources: recorded video files and live camera streams (RTSP/HTTP address or a webcam number)."""

from __future__ import annotations

import threading
import time
from urllib.parse import quote, unquote, urlsplit

import cv2
import numpy as np

Frame = tuple[np.ndarray, float]  # (image, timestamp in seconds since the source was opened)


RTSP_PORT = 554
# The part of a camera's stream address after the port depends on its brand.
STREAM_PATHS = {"Hikvision": "/Streaming/Channels/101", "Dahua": "/cam/realmonitor?channel=1&subtype=0"}


def is_live_source(source: str) -> bool:
    return "://" in source or source.isdigit()


def camera_stream_address(ip: str, user: str, password: str, path: str) -> str:
    """The RTSP address of a network camera. Characters such as @ : / in the user name or password are escaped."""
    login = quote(user, safe="")
    if password:
        login += ":" + quote(password, safe="")
    path = path if path.startswith("/") else "/" + path
    return f"rtsp://{login + '@' if login else ''}{ip}:{RTSP_PORT}{path}"


def camera_login(address: str) -> dict[str, str]:
    """The ip, user, password and path inside an RTSP address (empty strings for anything else)."""
    parts = urlsplit(address) if address.startswith("rtsp://") else urlsplit("")
    path = parts.path + ("?" + parts.query if parts.query else "")
    return {
        "ip": parts.hostname or "",
        "user": unquote(parts.username or ""),
        "password": unquote(parts.password or ""),
        "path": path,
    }


def grab_stream_frame(address: str, timeout_s: float = 8.0) -> np.ndarray | None:
    """One picture from a live stream, or None when it does not answer (wrong address, user name or password)."""
    timeout_ms = int(timeout_s * 1000)
    cap = cv2.VideoCapture(
        address, cv2.CAP_FFMPEG, [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms, cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms]
    )
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def _source_fps(cap: cv2.VideoCapture, fps_override: float | None) -> float:
    fps = fps_override or cap.get(cv2.CAP_PROP_FPS)
    return fps if fps and fps > 0 else 25.0


class FileSource:
    """Reads a recorded video frame by frame. Timestamps come from the frame number, so no frame is skipped."""

    live = False

    def __init__(self, path: str, fps_override: float | None, realtime: bool) -> None:
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Unable to open video: {path}")
        self.fps = _source_fps(self.cap, fps_override)
        self.realtime = realtime
        self.index = 0
        self.started = time.monotonic()

    def read(self) -> Frame | None:
        ok, frame = self.cap.read()
        if not ok:
            return None
        timestamp_s = self.index / self.fps
        self.index += 1
        if self.realtime:
            # Deliver frames no faster than they were recorded, the way a live camera would.
            delay = self.started + timestamp_s - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        return frame, timestamp_s

    def release(self) -> None:
        self.cap.release()


class StreamSource:
    """Reads a live camera on a background thread and keeps only the newest frame.

    When processing is slower than the camera, frames are skipped instead of piling up (the view would fall
    further and further behind). Timestamps are capture times, so speeds stay correct when frames are skipped.
    A stream that drops out is reopened.
    """

    live = True

    def __init__(self, address: str, fps_override: float | None, reconnect_delay_s: float = 3.0) -> None:
        self.address: str | int = int(address) if address.isdigit() else address
        self.cap = cv2.VideoCapture(self.address)
        if not self.cap.isOpened():
            raise RuntimeError(f"Unable to open stream: {address}")
        self.fps = _source_fps(self.cap, fps_override)
        self.reconnect_delay_s = reconnect_delay_s
        self.started = time.monotonic()
        self._latest: Frame | None = None
        self._new_frame = threading.Condition()
        self._closed = False
        self._thread = threading.Thread(target=self._read_forever, daemon=True)
        self._thread.start()

    def _read_forever(self) -> None:
        while not self._closed:
            ok, frame = self.cap.read()
            if not ok:
                self.cap.release()
                time.sleep(self.reconnect_delay_s)
                self.cap = cv2.VideoCapture(self.address)
                continue
            with self._new_frame:
                self._latest = (frame, time.monotonic() - self.started)
                self._new_frame.notify()

    def read(self, timeout_s: float = 1.0) -> Frame | None:
        """The newest frame not returned yet, or None when none arrived within the timeout (stream down)."""
        with self._new_frame:
            if self._latest is None:
                self._new_frame.wait(timeout_s)
            latest, self._latest = self._latest, None
        return latest

    def release(self) -> None:
        self._closed = True
        self._thread.join(timeout=5)
        self.cap.release()


def open_source(source: str, fps_override: float | None, realtime: bool) -> FileSource | StreamSource:
    if is_live_source(source):
        return StreamSource(source, fps_override)
    return FileSource(source, fps_override, realtime)
