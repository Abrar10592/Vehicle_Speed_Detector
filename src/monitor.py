"""Dashboard-side helpers: start/stop the processing service and index the evidence it saves."""

from __future__ import annotations

import csv
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil

from .config import list_camera_ids, read_cameras
from .live import camera_stop_file_name

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = "config/calibration.json"
RUNTIME_DIR = ROOT / "runtime"
LIVE_DIR = RUNTIME_DIR / "live"
# Each camera is processed by its own background process, so cameras are started and stopped one by one and a
# camera that fails does not stop the others. Per camera: <camera>.pid, <camera>.log and, from Start until Stop,
# <camera>.wanted (server.py starts a wanted camera again after a crash, a power cut or an update).
MONITOR_DIR = RUNTIME_DIR / "monitor"

# e.g. 19-14-34_lane_2_115.7kmh.mp4, or ..._2.mp4 for a second offender in the same second
CLIP_NAME = re.compile(r"^(?P<time>\d{2}-\d{2}-\d{2})_(?P<lane>.+)_(?P<speed>\d+(?:\.\d+)?)kmh(?:_\d+)?\.mp4$")
DATE_NAME = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def pid_file(camera_id: str) -> Path:
    return MONITOR_DIR / f"{camera_id}.pid"


def wanted_file(camera_id: str) -> Path:
    return MONITOR_DIR / f"{camera_id}.wanted"


def log_file(camera_id: str) -> Path:
    return MONITOR_DIR / f"{camera_id}.log"


def stop_file(camera_id: str) -> Path:
    return LIVE_DIR / camera_stop_file_name(camera_id)


def monitor_pid(camera_id: str) -> int | None:
    """PID of the process monitoring this camera, or None when the camera is not being monitored."""
    try:
        pid = int(pid_file(camera_id).read_text())
        process = psutil.Process(pid)
        command = process.cmdline()
        if process.status() != psutil.STATUS_ZOMBIE and "run_video.py" in " ".join(command) and camera_id in command:
            return pid
    except (OSError, ValueError, psutil.Error):
        pass
    return None


def running_cameras() -> list[str]:
    if not MONITOR_DIR.is_dir():
        return []
    return sorted(path.stem for path in MONITOR_DIR.glob("*.pid") if monitor_pid(path.stem) is not None)


def _cameras(camera_id: str | None, config_path: str) -> list[str]:
    return [camera_id] if camera_id else list_camera_ids(config_path)


def start_monitoring(camera_id: str | None = None, config_path: str = CONFIG_PATH) -> None:
    """Start monitoring one camera (or every camera in the config), publishing its live view to LIVE_DIR."""
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    MONITOR_DIR.mkdir(parents=True, exist_ok=True)
    for camera in _cameras(camera_id, config_path):
        wanted_file(camera).touch()
        if monitor_pid(camera) is not None:
            continue
        # The picture, status and stop request of this camera's previous run
        for old in (f"{camera}.jpg", f"{camera}.json", camera_stop_file_name(camera)):
            (LIVE_DIR / old).unlink(missing_ok=True)
        command = [
            sys.executable,
            "-u",
            str(ROOT / "run_video.py"),
            "--config",
            config_path,
            "--camera",
            camera,
            "--live-dir",
            str(LIVE_DIR),
            "--realtime",
        ]
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        with log_file(camera).open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
        pid_file(camera).write_text(str(process.pid))


def stop_monitoring(camera_id: str | None = None, config_path: str = CONFIG_PATH) -> None:
    """Ask one camera (or all) to stop; evidence clips being recorded are finished first."""
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    cameras = [camera_id] if camera_id else sorted({*running_cameras(), *_wanted_cameras()})
    for camera in cameras:
        wanted_file(camera).unlink(missing_ok=True)
        if monitor_pid(camera) is not None:
            stop_file(camera).touch()


def restart_monitoring(camera_id: str, config_path: str = CONFIG_PATH, wait_s: float = 90.0) -> None:
    """Stop a monitored camera and start it again, e.g. so it uses calibration that was just saved. This waits
    for the evidence clips being recorded to be finished, so it is best run in a background thread."""
    if monitor_pid(camera_id) is None:
        return
    stop_monitoring(camera_id)
    deadline = time.time() + wait_s
    while monitor_pid(camera_id) is not None and time.time() < deadline:
        time.sleep(0.5)
    if monitor_pid(camera_id) is not None:
        force_stop_monitoring(camera_id)
        time.sleep(1)
    if camera_id in list_camera_ids(config_path):  # not when the camera was removed from the calibration
        start_monitoring(camera_id, config_path)


def restart_in_background(camera_ids: list[str], config_path: str = CONFIG_PATH) -> None:
    for camera_id in camera_ids:
        threading.Thread(target=restart_monitoring, args=(camera_id, config_path), daemon=True).start()


def _wanted_cameras() -> list[str]:
    return [path.stem for path in MONITOR_DIR.glob("*.wanted")] if MONITOR_DIR.is_dir() else []


def stop_requested(camera_id: str) -> bool:
    return stop_file(camera_id).exists()


def force_stop_monitoring(camera_id: str | None = None) -> None:
    cameras = [camera_id] if camera_id else running_cameras()
    for camera in cameras:
        wanted_file(camera).unlink(missing_ok=True)
        pid = monitor_pid(camera)
        if pid is not None:
            try:
                for process in [psutil.Process(pid), *psutil.Process(pid).children(recursive=True)]:
                    process.kill()
            except psutil.Error:
                pass


def log_tail(camera_id: str, lines: int = 30) -> str:
    try:
        return "\n".join(log_file(camera_id).read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


def evidence_roots(config_path: str = CONFIG_PATH) -> list[Path]:
    roots: list[Path] = []
    for camera in read_cameras(config_path):
        root = Path(camera.get("evidence_root") or "evidence")
        root = root if root.is_absolute() else ROOT / root
        if root not in roots:
            roots.append(root)
    return roots


def _speed_limits_by_clip(root: Path) -> dict[Path, float]:
    limits: dict[Path, float] = {}
    try:
        with (root / "speed_events.csv").open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("clip_path"):
                    clip = Path(row["clip_path"])
                    limits[(clip if clip.is_absolute() else ROOT / clip).resolve()] = float(row["speed_limit_kmph"])
    except (OSError, ValueError, KeyError):
        pass
    return limits


def list_clips(roots: list[Path]) -> list[dict]:
    """Every evidence clip under <root>/<date>/<camera>/, newest first."""
    clips: list[dict] = []
    for root in roots:
        if not root.is_dir():
            continue
        limits = _speed_limits_by_clip(root)
        for date_dir in root.iterdir():
            if not (date_dir.is_dir() and DATE_NAME.match(date_dir.name)):
                continue
            for camera_dir in date_dir.iterdir():
                if not camera_dir.is_dir():
                    continue
                for clip in camera_dir.glob("*.mp4"):
                    match = CLIP_NAME.match(clip.name)
                    if match is None:
                        continue
                    clips.append(
                        {
                            "date": date_dir.name,
                            "time": match["time"].replace("-", ":"),
                            "camera": camera_dir.name,
                            "lane": match["lane"],
                            "speed_kmph": float(match["speed"]),
                            "limit_kmph": limits.get(clip.resolve()),
                            "path": str(clip),
                        }
                    )
    clips.sort(key=lambda c: (c["date"], c["time"]), reverse=True)
    return clips
