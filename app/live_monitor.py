import time

import streamlit as st

from src.config import list_camera_ids
from src.live import read_frame, read_status
from src.monitor import (
    CONFIG_PATH,
    LIVE_DIR,
    force_stop_monitoring,
    log_tail,
    monitor_pid,
    start_monitoring,
    stop_monitoring,
    stop_requested,
)

# A running camera that has not reported for this long has lost its video.
STALE_AFTER_S = 5.0
STATE_BADGES = {
    "starting": ("Starting", "blue"),
    "running": ("Live", "green"),
    "reconnecting": ("Reconnecting", "orange"),
    "stopping": ("Stopping", "orange"),
    "finished": ("Video ended", "gray"),
    "stopped": ("Stopped", "gray"),
    "error": ("Error", "red"),
}

st.title("Live monitor")


def camera_controls(camera_id: str, running: bool) -> None:
    if running and stop_requested(camera_id):
        st.button(
            "Force stop",
            key=f"force_{camera_id}",
            icon=":material/dangerous:",
            help="Ends this camera at once; an evidence clip being recorded is lost.",
            on_click=force_stop_monitoring,
            args=(camera_id,),
        )
    elif running:
        st.button("Stop", key=f"stop_{camera_id}", icon=":material/stop:", on_click=stop_monitoring, args=(camera_id,))
    else:
        st.button(
            "Start",
            key=f"start_{camera_id}",
            type="primary",
            icon=":material/play_arrow:",
            on_click=start_monitoring,
            args=(camera_id,),
        )


@st.fragment(run_every=0.4)
def live_view() -> None:
    try:
        cameras = list_camera_ids(CONFIG_PATH)
    except (OSError, ValueError) as exc:
        st.error(f"Cannot read the calibration file {CONFIG_PATH}: {exc}")
        return
    running = {camera_id: monitor_pid(camera_id) is not None for camera_id in cameras}

    with st.container(horizontal=True, vertical_alignment="center"):
        count = sum(running.values())
        if count == len(cameras) and cameras:
            st.success("Monitoring all cameras", icon=":material/radio_button_checked:")
        elif count:
            st.success(f"Monitoring {count} of {len(cameras)} cameras", icon=":material/radio_button_checked:")
        else:
            st.info("Monitoring is stopped", icon=":material/pause_circle:")
        if count < len(cameras):
            st.button("Start all", icon=":material/play_arrow:", on_click=start_monitoring)
        if count:
            st.button("Stop all", icon=":material/stop:", on_click=stop_monitoring)

    last_frames = st.session_state.setdefault("live_frames", {})
    columns = st.columns(2)
    for index, camera_id in enumerate(cameras):
        with columns[index % 2].container(border=True):
            status = read_status(LIVE_DIR, camera_id) or {}
            state = status.get("state")
            if running[camera_id] and stop_requested(camera_id):
                state = "stopping"
            elif state in ("starting", "running", "reconnecting") and not running[camera_id]:
                state = "stopped"  # the process ended without reporting (e.g. force stop)
            elif state == "running" and time.time() - status.get("updated", 0) > STALE_AFTER_S:
                state = "reconnecting"
            label, color = STATE_BADGES.get(state, ("Not started", "gray"))

            with st.container(horizontal=True, vertical_alignment="center"):
                st.markdown(f"**{camera_id}**")
                st.badge(label, color=color)
                if state == "running":
                    st.caption(f"{status.get('fps', 0)} fps")
                st.caption(f"Speeding: {status.get('speeders', 0)}")
                st.space("stretch")
                camera_controls(camera_id, running[camera_id])

            frame = read_frame(LIVE_DIR, camera_id) if running[camera_id] or state else None
            if frame is not None:
                last_frames[camera_id] = frame
            frame = frame or last_frames.get(camera_id)
            if state == "error":
                st.error(status.get("error", "Processing failed."))
            elif frame is not None:
                st.image(frame, width="stretch")
            else:
                st.caption("No picture yet. Press Start to monitor this camera.")

            if not running[camera_id] and log_tail(camera_id):
                with st.expander("Last monitoring log"):
                    st.code(log_tail(camera_id), language=None)


live_view()
