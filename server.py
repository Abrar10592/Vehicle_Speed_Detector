"""Keeps the Vehicle Speed Detector running on a server nobody is sitting at. Start it with Start.bat.

    python server.py                       run the app and keep it running
    python server.py --install-autostart   also start it whenever this Windows user logs in
    python server.py --remove-autostart

While it runs:
- the app (http://localhost:8501) is started, and started again if it ends;
- monitoring that was on (Start monitoring pressed, Stop monitoring not pressed) is started again after a
  crash, a power cut or an update;
- the app is restarted after an update, so the new version is the one in use.

For it to survive a restart of the computer, Windows must log this user in automatically.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / "runtime"
SERVER_FILE = RUNTIME / "server.json"  # this process and the app it started
APP_LOG = RUNTIME / "app.log"
RESTART_APP_FILE = RUNTIME / "restart_app"
UPDATE_STATUS_FILE = RUNTIME / "update_status.json"
MONITOR_PID_FILE = RUNTIME / "monitor.pid"
MONITOR_WANTED_FILE = RUNTIME / "monitor.wanted"
LIVE_DIR = RUNTIME / "live"
AUTOSTART_NAME = "Vehicle Speed Detector.cmd"
NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
CHECK_EVERY_S = 3.0
RETRY_AFTER_S = 30.0  # between two attempts to start something that keeps ending


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _running(pid: int | None, script: str) -> bool:
    """Whether process `pid` is alive and is the program we think it is (PIDs get reused)."""
    try:
        process = psutil.Process(int(pid))
        return process.status() != psutil.STATUS_ZOMBIE and script in " ".join(process.cmdline())
    except (TypeError, ValueError, psutil.Error):
        return False


def _kill_app(pid: int) -> None:
    """End the app and what it started, except the monitoring service: that one keeps measuring."""
    try:
        parent = psutil.Process(pid)
        processes = [*parent.children(recursive=True), parent]
    except psutil.Error:
        return
    for process in processes:
        try:
            if "run_video.py" not in " ".join(process.cmdline()):
                process.kill()
        except psutil.Error:
            pass


def start_app(port: int) -> subprocess.Popen:
    command = [
        sys.executable, "-m", "streamlit", "run", str(ROOT / "streamlit_app.py"),
        "--server.port", str(port), "--server.headless", "true",
    ]  # fmt: skip
    with APP_LOG.open("a", encoding="utf-8") as log:
        return subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, creationflags=NO_WINDOW)


def monitoring_should_restart() -> bool:
    """Monitoring was on and its process is gone without having finished or been stopped."""
    if not MONITOR_WANTED_FILE.exists() or _read_json(UPDATE_STATUS_FILE).get("state") == "running":
        return False
    try:
        pid = int(MONITOR_PID_FILE.read_text())
    except (OSError, ValueError):
        pid = None
    if _running(pid, "run_video.py") or (LIVE_DIR / "STOP").exists():
        return False
    # A run over recorded videos ends by itself ("finished"); only an interrupted one is started again.
    states = [_read_json(path).get("state") for path in LIVE_DIR.glob("*.json")] if LIVE_DIR.is_dir() else []
    return not states or any(state not in ("finished", "stopped") for state in states)


def start_monitoring() -> None:
    # In a new process, so that the version installed by the latest update is the one that runs.
    subprocess.run(
        [sys.executable, "-c", "from src.monitor import start_monitoring; start_monitoring()"],
        cwd=ROOT,
        capture_output=True,
        creationflags=NO_WINDOW,
    )


def serve(port: int, open_browser: bool) -> int:
    RUNTIME.mkdir(exist_ok=True)
    other = _read_json(SERVER_FILE).get("server_pid")
    if other != os.getpid() and _running(other, "server.py"):
        print("The Vehicle Speed Detector is already running on this computer.")
        return 0

    left_over = _read_json(SERVER_FILE).get("app_pid")  # the app of a server whose window was closed
    if _running(left_over, "streamlit"):
        _kill_app(int(left_over))

    app: subprocess.Popen | None = None
    app_started = monitor_started = 0.0
    print(f"Vehicle Speed Detector: http://localhost:{port}   (close this window to stop the app)")
    try:
        while True:
            now = time.time()
            if RESTART_APP_FILE.exists() and _read_json(UPDATE_STATUS_FILE).get("state") != "running":
                RESTART_APP_FILE.unlink(missing_ok=True)
                if app is not None:
                    print(f"{time.strftime('%H:%M:%S')}  restarting the app after an update")
                    _kill_app(app.pid)
                    app.wait()
                    app_started = 0.0
            if (app is None or app.poll() is not None) and now - app_started > RETRY_AFTER_S:
                print(f"{time.strftime('%H:%M:%S')}  starting the app")
                app, app_started = start_app(port), now
                SERVER_FILE.write_text(json.dumps({"server_pid": os.getpid(), "app_pid": app.pid, "port": port}))
                if open_browser:
                    open_browser = False
                    webbrowser.open(f"http://localhost:{port}")
            if now - monitor_started > RETRY_AFTER_S and monitoring_should_restart():
                print(f"{time.strftime('%H:%M:%S')}  starting monitoring again")
                monitor_started = now
                start_monitoring()
            time.sleep(CHECK_EVERY_S)
    except KeyboardInterrupt:
        return 0
    finally:
        if app is not None:
            _kill_app(app.pid)
        SERVER_FILE.unlink(missing_ok=True)


def autostart_file() -> Path:
    return Path(os.environ["APPDATA"]) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup" / AUTOSTART_NAME


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Vehicle Speed Detector app and keep it running")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--no-browser", action="store_true", help="do not open the app in the browser")
    parser.add_argument("--install-autostart", action="store_true", help="start whenever this Windows user logs in")
    parser.add_argument("--remove-autostart", action="store_true")
    args = parser.parse_args()
    if args.install_autostart:
        autostart_file().write_text(f'@start "Vehicle Speed Detector" /min "{ROOT / "Start.bat"}"\n', encoding="ascii")
        print(f"Installed: {autostart_file()}")
        return 0
    if args.remove_autostart:
        autostart_file().unlink(missing_ok=True)
        print("Removed.")
        return 0
    return serve(args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    raise SystemExit(main())
