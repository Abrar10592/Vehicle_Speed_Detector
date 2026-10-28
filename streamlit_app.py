import streamlit as st

st.set_page_config(page_title="Vehicle Speed Detector", page_icon=":material/speed:", layout="wide")

page = st.navigation(
    {
        "": [st.Page("app/home.py", title="Home", icon=":material/home:", default=True)],
        "Setup": [st.Page("app/calibration_dashboard.py", title="Calibration", icon=":material/tune:")],
        "Operator": [
            st.Page("app/live_monitor.py", title="Live monitor", icon=":material/videocam:"),
            st.Page("app/evidence_browser.py", title="Evidence", icon=":material/video_library:"),
        ],
    },
    position="top",
)
page.run()
