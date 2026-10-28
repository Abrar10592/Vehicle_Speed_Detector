"""Dashboard-side helpers: start/stop the processing service and index the evidence it saves."""

from __future__ import annotations

import csv
import os
import re
import subprocess
import sys
from pathlib import Path

import psutil

from .config import read_cameras
from .live import STOP_FILE_NAME

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = "config/calibration.json"
RUNTIME_DIR = ROOT / "runtime"
LIVE_DIR = RUNTIME_DIR / "live"
PID_FILE = RUNTIME_DIR / "monitor.pid"
LOG_FILE = RUNTIME_DIR / "monitor.log"

# e.g. 19-14-34_lane_2_115.7kmh.mp4, or ..._2.mp4 for a second offender in the same second
CLIP_NAME = re.compile(r"^(?P<time>\d{2}-\d{2}-\d{2})_(?P<lane>.+)_(?P<speed>\d+(?:\.\d+)?)kmh(?:_\d+)?\.mp4$")
DATE_NAME = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def monitor_pid() -> int | None:
    """PID of the running processing service, or None when it is not running."""
    try:
        pid = int(PID_FILE.read_text())
        process = psutil.Process(pid)
        if process.status() != psutil.STATUS_ZOMBIE and "run_video.py" in " ".join(process.cmdline()):
            return pid
    except (OSError, ValueError, psutil.Error):
        pass
    return None


def start_monitoring(config_path: str = CONFIG_PATH) -> None:
    """Start processing every camera in the config in the background, publishing live views to LIVE_DIR."""
    if monitor_pid() is not None:
        return
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    for old in LIVE_DIR.iterdir():  # frames, statuses and the STOP request of the previous run
        old.unlink(missing_ok=True)
    command = [
        sys.executable,
        "-u",
        str(ROOT / "run_video.py"),
        "--config",
        config_path,
        "--camera",
        "all",
        "--live-dir",
        str(LIVE_DIR),
        "--realtime",
    ]
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    with LOG_FILE.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
    PID_FILE.write_text(str(process.pid))


def stop_monitoring() -> None:
    """Ask the service to stop; it finishes the evidence clips being recorded first."""
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    (LIVE_DIR / STOP_FILE_NAME).touch()


def stop_requested() -> bool:
    return (LIVE_DIR / STOP_FILE_NAME).exists()


def force_stop_monitoring() -> None:
    pid = monitor_pid()
    if pid is not None:
        for process in [psutil.Process(pid), *psutil.Process(pid).children(recursive=True)]:
            process.kill()


def log_tail(lines: int = 30) -> str:
    try:
        return "\n".join(LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
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
