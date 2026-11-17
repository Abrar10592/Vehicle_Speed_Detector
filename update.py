"""Update this computer to the newest released version from GitHub.

    python update.py --check      show the installed and the newest released version
    python update.py --apply      install the newest released version (what the app's Update page runs)
    python update.py --repair     reinstall the Python packages and test the installed version
    python update.py --undo       go back to the version that was installed before the last update

Update.bat runs --apply --repair, for when the app itself does not open.

A released version is a Git tag named like v1.2 that has been pushed to GitHub; work that is only on the main
branch is not installed. An update that fails its self-test (selftest.py) is removed again, so a computer
nobody is sitting at is left on a version that works. Only the Python standard library is used here, so this
file still runs when a package is broken.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / "runtime"
STATUS_FILE = RUNTIME / "update_status.json"
LOG_FILE = RUNTIME / "update.log"
RESTART_APP_FILE = RUNTIME / "restart_app"  # server.py restarts the app when it sees this
BACKUP_DIR = RUNTIME / "backup"
CONFIG = ROOT / "config" / "calibration.json"
LOCK_FILE = ROOT / "requirements.lock.txt"
LIVE_DIR = RUNTIME / "live"
MONITOR_PID_FILE = RUNTIME / "monitor.pid"
RELEASE_TAGS = "v*"
NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


class UpdateError(Exception):
    pass


def log(message: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {message}"
    print(line, flush=True)
    RUNTIME.mkdir(exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def write_status(state: str, message: str, **fields) -> None:
    """state: running, success, up_to_date, rolled_back or failed."""
    RUNTIME.mkdir(exist_ok=True)
    status = {**read_status(), "state": state, "message": message, "time": time.time(), **fields}
    STATUS_FILE.write_text(json.dumps(status), encoding="utf-8")


def read_status() -> dict:
    try:
        return json.loads(STATUS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def find_git() -> str:
    found = shutil.which("git")
    if found:
        return found
    # Git for Windows is not always on the PATH of the program that starts this.
    folders = [os.environ.get("ProgramFiles", ""), os.environ.get("LOCALAPPDATA", "") + "\\Programs", "C:\\", "D:\\"]
    for folder in folders:
        candidate = Path(folder) / "Git" / "cmd" / "git.exe"
        if folder and candidate.exists():
            return str(candidate)
    raise UpdateError("Git is not installed on this computer.")


def git(*args: str) -> str:
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}  # never wait for a password nobody will type
    result = subprocess.run(
        [find_git(), *args], cwd=ROOT, capture_output=True, text=True, env=env, creationflags=NO_WINDOW
    )
    if result.returncode != 0:
        raise UpdateError(f"git {' '.join(args)}: {(result.stderr or result.stdout).strip()}")
    return result.stdout.strip()


def installed_version() -> str:
    return git("describe", "--tags", "--always")


def check(fetch: bool = True) -> dict:
    """The installed version, the newest released one, and whether it can be installed."""
    if fetch:
        try:
            git("fetch", "--tags", "--quiet", "origin")
        except UpdateError as exc:
            raise UpdateError(f"Cannot reach GitHub. Check the internet connection. ({exc})") from exc
    info = {"installed": installed_version(), "latest": None, "available": False, "changes": [], "note": ""}
    tags = git("tag", "--list", RELEASE_TAGS, "--sort=-v:refname").splitlines()
    if not tags:
        info["note"] = "No version has been released yet."
        return info
    info["latest"] = latest = tags[0]
    head, target = git("rev-parse", "HEAD"), git("rev-list", "-n", "1", latest)
    if head == target:
        info["note"] = "This computer has the newest released version."
    elif git("merge-base", head, target) != head:
        # e.g. a development computer with work that is newer than the last release
        info["note"] = f"This computer has changes that are not part of {latest}; it is not updated automatically."
    else:
        info["available"] = True
        info["changes"] = git("log", "--format=%s", f"{head}..{target}").splitlines()
    return info


def _process_alive(pid: int) -> bool:
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    out = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True, creationflags=NO_WINDOW
    ).stdout
    return str(pid) in out


def _monitor_pid() -> int | None:
    try:
        pid = int(MONITOR_PID_FILE.read_text())
    except (OSError, ValueError):
        return None
    return pid if _process_alive(pid) else None


def stop_monitoring(wait_s: float = 90.0) -> bool:
    """Stop the monitoring service for the update. Returns whether it was running (it is started again after)."""
    pid = _monitor_pid()
    if pid is None:
        return False
    log("Stopping monitoring (clips being recorded are saved first)...")
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    (LIVE_DIR / "STOP").touch()
    deadline = time.time() + wait_s
    while time.time() < deadline and _process_alive(pid):
        time.sleep(1)
    if _process_alive(pid):
        log("Monitoring did not stop by itself; ending it.")
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, creationflags=NO_WINDOW)
    return True


def start_monitoring() -> None:
    log("Starting monitoring again...")
    result = subprocess.run(
        [sys.executable, "-c", "from src.monitor import start_monitoring; start_monitoring()"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        creationflags=NO_WINDOW,
    )
    if result.returncode != 0:
        log(f"Monitoring could not be started: {result.stderr.strip()[-400:]}")


def backup_calibration() -> Path | None:
    if not CONFIG.exists():
        return None
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    backup = BACKUP_DIR / f"calibration-{time.strftime('%Y%m%d-%H%M%S')}.json"
    shutil.copy2(CONFIG, backup)
    return backup


def restore_calibration(backup: Path | None) -> None:
    """The cameras set up on this computer are never replaced by an update."""
    if backup is not None and (not CONFIG.exists() or CONFIG.read_bytes() != backup.read_bytes()):
        CONFIG.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(backup, CONFIG)
        log("Restored this computer's calibration.")


def install_packages() -> None:
    log("Installing the Python packages (this can take several minutes)...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "-r", str(LOCK_FILE)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        creationflags=NO_WINDOW,
    )
    if result.returncode != 0:
        raise UpdateError(f"Installing packages failed: {(result.stderr or result.stdout).strip()[-600:]}")


def run_selftest() -> None:
    log("Testing the installed version...")
    try:
        result = subprocess.run(
            [sys.executable, str(ROOT / "selftest.py")],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=900,
            creationflags=NO_WINDOW,
        )
    except subprocess.TimeoutExpired as exc:
        raise UpdateError("The self-test did not finish within 15 minutes.") from exc
    if result.returncode != 0:
        lines = (result.stdout + result.stderr).strip().splitlines()
        failed = next((line for line in lines if line.startswith("SELF-TEST FAILED")), "\n".join(lines[-5:]))
        raise UpdateError(failed)
    log("Self-test passed.")


def _lock_text() -> str:
    return LOCK_FILE.read_text(encoding="utf-8") if LOCK_FILE.exists() else ""


def apply(repair: bool = False) -> int:
    """Install the newest released version; on any failure go back to the version that was running."""
    write_status("running", "Checking for a new version...")
    try:
        info = check()
    except UpdateError as exc:
        log(str(exc))
        write_status("failed", str(exc))
        return 1
    log(f"Installed: {info['installed']}. Newest released: {info['latest'] or 'none'}.")
    if not info["available"] and not repair:
        log(info["note"])
        write_status("up_to_date", info["note"], installed=info["installed"])
        return 0

    previous = git("rev-parse", "HEAD")
    old_lock = _lock_text()
    was_monitoring = stop_monitoring()
    backup = backup_calibration()
    code, state = 0, "success"
    try:
        if info["available"]:
            write_status("running", f"Installing {info['latest']}...")
            log(f"Installing {info['latest']}...")
            git("merge", "--ff-only", info["latest"])
            restore_calibration(backup)
        if repair or _lock_text() != old_lock:
            write_status("running", "Installing the Python packages...")
            install_packages()
        write_status("running", "Testing the new version...")
        run_selftest()
        message = f"Updated to {installed_version()}." if info["available"] else "The installed version works."
    except Exception as exc:  # whatever went wrong, this computer must end up on a working version
        code, state = 1, "failed"
        message = f"{exc}"
        log(f"FAILED: {exc}")
        if git("rev-parse", "HEAD") != previous:
            try:
                log(f"Going back to {info['installed']}...")
                git("reset", "--keep", previous)
                if _lock_text() != old_lock or repair:
                    install_packages()
                state = "rolled_back"
                message = f"{info['latest']} was not installed because it failed: {exc} This computer is back on {info['installed']}."
            except UpdateError as undo_exc:
                message = f"The update failed ({exc}) and going back failed too ({undo_exc}). Run Update.bat."
        restore_calibration(backup)
    log(message)
    if was_monitoring:
        start_monitoring()
    if state == "success" and info["available"]:
        RESTART_APP_FILE.touch()  # the app is still running the old version until it is restarted
        write_status(state, message, installed=installed_version(), previous=previous)  # what --undo goes back to
    else:
        write_status(state, message, installed=installed_version())
    return code


def undo() -> int:
    """Go back to the version that was installed before the last update."""
    previous = read_status().get("previous")
    if not previous:
        log("There is no earlier update to undo.")
        return 1
    was_monitoring = stop_monitoring()
    backup = backup_calibration()
    old_lock = _lock_text()
    try:
        git("reset", "--keep", previous)
        restore_calibration(backup)
        if _lock_text() != old_lock:
            install_packages()
    except UpdateError as exc:
        log(f"FAILED: {exc}")
        write_status("failed", str(exc))
        return 1
    finally:
        if was_monitoring:
            start_monitoring()
    message = f"Went back to {installed_version()}."
    log(message)
    RESTART_APP_FILE.touch()
    write_status("success", message, installed=installed_version(), previous=None)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Update the Vehicle Speed Detector to the newest released version")
    parser.add_argument("--check", action="store_true", help="only show the installed and the newest released version")
    parser.add_argument("--apply", action="store_true", help="install the newest released version")
    parser.add_argument("--repair", action="store_true", help="reinstall the Python packages and test the installed version")
    parser.add_argument("--undo", action="store_true", help="go back to the version before the last update")
    args = parser.parse_args()
    try:
        if args.undo:
            return undo()
        if args.apply or args.repair:
            return apply(repair=args.repair)
        info = check()
        print(f"Installed version:      {info['installed']}")
        print(f"Newest released version: {info['latest'] or 'none'}")
        print(info["note"] or f"An update is available ({len(info['changes'])} changes):")
        for change in info["changes"]:
            print(f"  - {change}")
        return 0
    except UpdateError as exc:
        log(str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
