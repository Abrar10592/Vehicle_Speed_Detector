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
    "finished": ("Video ended", "gray"),
    "stopped": ("Stopped", "gray"),
    "error": ("Error", "red"),
}

st.title("Live monitor")


@st.fragment(run_every=0.4)
def live_view() -> None:
    running = monitor_pid() is not None

    with st.container(horizontal=True, vertical_alignment="center"):
        if running and stop_requested():
            st.warning("Stopping: saving the evidence clips being recorded...", icon=":material/hourglass_top:")
            st.button("Force stop", icon=":material/dangerous:", on_click=force_stop_monitoring)
        elif running:
            st.success("Monitoring all cameras", icon=":material/radio_button_checked:")
            st.button("Stop monitoring", icon=":material/stop:", on_click=stop_monitoring)
        else:
            st.info("Monitoring is stopped", icon=":material/pause_circle:")
            st.button("Start monitoring", type="primary", icon=":material/play_arrow:", on_click=start_monitoring)

    try:
        cameras = list_camera_ids(CONFIG_PATH)
    except (OSError, ValueError) as exc:
        st.error(f"Cannot read the calibration file {CONFIG_PATH}: {exc}")
        return

    last_frames = st.session_state.setdefault("live_frames", {})
    columns = st.columns(2)
    for index, camera_id in enumerate(cameras):
        with columns[index % 2].container(border=True):
            status = read_status(LIVE_DIR, camera_id) or {}
            state = status.get("state")
            if state in ("starting", "running", "reconnecting") and not running:
                state = "stopped"  # the service ended without reporting (e.g. force stop)
            elif state == "running" and time.time() - status.get("updated", 0) > STALE_AFTER_S:
                state = "reconnecting"
            label, color = STATE_BADGES.get(state, ("Not started", "gray"))

            with st.container(horizontal=True, vertical_alignment="center"):
                st.markdown(f"**{camera_id}**")
                st.badge(label, color=color)
                if state == "running":
                    st.caption(f"{status.get('fps', 0)} fps")
                st.caption(f"Speeding: {status.get('speeders', 0)}")

            frame = read_frame(LIVE_DIR, camera_id)
            if frame is not None:
                last_frames[camera_id] = frame
            frame = frame or last_frames.get(camera_id)
            if state == "error":
                st.error(status.get("error", "Processing failed."))
            elif frame is not None:
                st.image(frame, width="stretch")
            else:
                st.caption("No picture yet. Start monitoring to see this camera.")

    if not running and log_tail():
        with st.expander("Last monitoring log"):
            st.code(log_tail(), language=None)


live_view()
