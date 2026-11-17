import json
import subprocess
import sys
import time

import streamlit as st

import update as updater

RUNNING_TOO_LONG_S = 45 * 60  # a status still "running" after this is from an update that was interrupted


def supervised() -> bool:
    """Whether server.py (Start.bat) is running: it restarts the app after an update."""
    try:
        pid = json.loads((updater.RUNTIME / "server.json").read_text(encoding="utf-8"))["server_pid"]
        return updater._process_alive(int(pid))
    except (OSError, ValueError, KeyError):
        return False


def update_running(status: dict) -> bool:
    return status.get("state") == "running" and time.time() - status.get("time", 0) < RUNNING_TOO_LONG_S


def start_update() -> None:
    updater.write_status("running", "Starting the update...")
    flags = (subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP) if sys.platform == "win32" else 0
    subprocess.Popen(
        [sys.executable, str(updater.ROOT / "update.py"), "--apply"],
        cwd=updater.ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=flags,
    )
    st.session_state.pop("update_info", None)


def check_for_updates() -> None:
    try:
        st.session_state.update_info = updater.check()
    except updater.UpdateError as exc:
        st.session_state.update_info = {"error": str(exc)}


@st.fragment(run_every=2)
def progress() -> None:
    status = updater.read_status()
    if not update_running(status):
        st.rerun(scope="app")
    st.info(status.get("message", "Updating..."), icon=":material/hourglass_top:")
    try:
        st.code("\n".join(updater.LOG_FILE.read_text(encoding="utf-8").splitlines()[-12:]), language=None)
    except OSError:
        pass


st.title("Update")
st.caption(
    "Installs the newest released version from GitHub. The cameras and calibration on this computer are kept. "
    "A version that does not pass its self-test is removed again and the current one stays."
)

status = updater.read_status()
try:
    st.metric("Installed version", updater.installed_version(), border=True)
except updater.UpdateError as exc:
    st.error(str(exc))

if update_running(status):
    progress()
    st.stop()

if status.get("state") in ("success", "rolled_back", "failed"):
    when = time.strftime("%d %b %Y %H:%M", time.localtime(status.get("time", 0)))
    show = {"success": st.success, "rolled_back": st.warning, "failed": st.error}[status["state"]]
    show(f"Last update ({when}): {status.get('message', '')}")
if updater.RESTART_APP_FILE.exists():
    if supervised():
        st.info("The app restarts in a few seconds to use the new version.", icon=":material/restart_alt:")
    else:
        st.warning("Close the app and start it again with Start.bat to use the new version.")

st.button("Check for updates", icon=":material/refresh:", on_click=check_for_updates)
info = st.session_state.get("update_info")
if info and info.get("error"):
    st.error(info["error"])
elif info and info["available"]:
    st.success(f"Version {info['latest']} is available.", icon=":material/new_releases:")
    st.markdown("\n".join(f"- {change}" for change in info["changes"]))
    st.caption(
        "Updating takes a few minutes. Monitoring pauses while it runs and starts again by itself; "
        "clips being recorded are saved first."
    )
    st.button("Update now", type="primary", icon=":material/download:", on_click=start_update)
elif info:
    st.info(info["note"])

with st.expander("If the app does not open"):
    st.markdown(
        "Double-click **Update.bat** in the program's folder. It installs the newest released version, "
        "reinstalls the Python packages and tests the result, without the app. "
        "Then start the app with **Start.bat**."
    )
