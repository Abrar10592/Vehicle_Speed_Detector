import os
from pathlib import Path

import pandas as pd
import streamlit as st

from src.monitor import CONFIG_PATH, evidence_roots, list_clips

st.title("Evidence")

try:
    roots = evidence_roots(CONFIG_PATH)
except (OSError, ValueError) as exc:
    st.error(f"Cannot read the calibration file {CONFIG_PATH}: {exc}")
    st.stop()


def open_folder(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    os.startfile(path)  # Windows Explorer, on the computer running this app


with st.container(horizontal=True):
    for root in roots:
        if hasattr(os, "startfile"):
            st.button(
                f"Open folder: {root.name}",
                icon=":material/folder_open:",
                on_click=open_folder,
                args=(root,),
                key=f"open_{root}",
            )
        else:
            st.caption(f"Evidence folder: {root}")

clips_tab, log_tab = st.tabs(["Clips", "Speed log"])

with clips_tab:
    clips = list_clips(roots)
    if not clips:
        st.info("No evidence clips yet. Clips of speeding vehicles appear here once monitoring has caught one.")
    else:
        dates = sorted({clip["date"] for clip in clips}, reverse=True)
        filter_date, filter_camera = st.columns(2)
        day = filter_date.selectbox("Date", dates)
        day_cameras = sorted({clip["camera"] for clip in clips if clip["date"] == day})
        camera = filter_camera.selectbox("Camera", ["All cameras", *day_cameras])
        shown = [c for c in clips if c["date"] == day and camera in ("All cameras", c["camera"])]

        table, player = st.columns([3, 2])
        with table:
            selection = st.dataframe(
                pd.DataFrame(shown)[["time", "camera", "lane", "speed_kmph", "limit_kmph"]],
                column_config={
                    "time": "Time",
                    "camera": "Camera",
                    "lane": "Lane",
                    "speed_kmph": st.column_config.NumberColumn("Speed", format="%.1f km/h"),
                    "limit_kmph": st.column_config.NumberColumn("Limit", format="%.0f km/h"),
                },
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row",
                key=f"clips_{day}_{camera}",
            )
            st.caption(f"{len(shown)} speeding vehicle(s). Select a row to play its clip.")
        with player:
            rows = selection.selection.rows
            clip = shown[rows[0] if rows else 0]
            st.markdown(f"**{clip['camera']} · {clip['lane']} · {clip['speed_kmph']:.1f} km/h** at {clip['time']}")
            st.video(clip["path"])
            st.download_button(
                "Download clip",
                data=Path(clip["path"]).read_bytes(),
                file_name=Path(clip["path"]).name,
                mime="video/mp4",
                icon=":material/download:",
            )

with log_tab:
    logs = [pd.read_csv(root / "speed_events.csv") for root in roots if (root / "speed_events.csv").is_file()]
    if not logs:
        st.info("No speeds measured yet.")
    else:
        log = pd.concat(logs, ignore_index=True)
        if "camera_id" in log:
            cameras = st.multiselect("Cameras", sorted(log["camera_id"].astype(str).unique()))
            if cameras:
                log = log[log["camera_id"].astype(str).isin(cameras)]
        log["speeding"] = log["speed_kmph"] > log["speed_limit_kmph"]
        st.dataframe(
            log.drop(columns=[c for c in ("bbox", "track_path") if c in log]),
            column_config={
                "speed_kmph": st.column_config.NumberColumn("Speed", format="%.1f km/h"),
                "speed_limit_kmph": st.column_config.NumberColumn("Limit", format="%.0f km/h"),
                "timestamp_s": st.column_config.NumberColumn("Video time", format="%.1f s"),
            },
            hide_index=True,
        )
        st.download_button(
            "Download CSV",
            data=log.to_csv(index=False).encode("utf-8"),
            file_name="speed_events.csv",
            mime="text/csv",
            icon=":material/download:",
        )
