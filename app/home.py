from datetime import date

import streamlit as st

from src.config import list_camera_ids
from src.monitor import CONFIG_PATH, evidence_roots, list_clips, monitor_pid

st.title("Vehicle Speed Detector")
st.caption("Speed measurement and evidence capture from CCTV cameras and recorded videos.")

try:
    cameras = list_camera_ids(CONFIG_PATH)
    today = date.today().isoformat()
    offenders_today = sum(1 for clip in list_clips(evidence_roots()) if clip["date"] == today)
except (OSError, ValueError):
    cameras, offenders_today = [], 0

with st.container(horizontal=True):
    st.metric("Cameras", len(cameras), border=True)
    st.metric("Monitoring", "Running" if monitor_pid() else "Stopped", border=True)
    st.metric("Speeding vehicles today", offenders_today, border=True)

setup, operator = st.columns(2)
with setup.container(border=True, height="stretch"):
    st.subheader(":material/tune: Setup")
    st.write(
        "Add and name cameras, choose each camera's video file or live stream, draw the lanes and "
        "measurement lines, and set speed limits."
    )
    st.page_link("app/calibration_dashboard.py", label="Open calibration", icon=":material/arrow_forward:")

with operator.container(border=True, height="stretch"):
    st.subheader(":material/videocam: Operator")
    st.write("Watch every camera live with vehicle speeds on screen, and review the clips of speeding vehicles.")
    st.page_link("app/live_monitor.py", label="Live monitor", icon=":material/arrow_forward:")
    st.page_link("app/evidence_browser.py", label="Evidence", icon=":material/arrow_forward:")
