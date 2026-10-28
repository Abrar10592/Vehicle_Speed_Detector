from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas  # type: ignore[import-not-found]
from PIL import Image


DEFAULT_OUTPUT_CONFIG = "config/calibration.json"
DEFAULT_PREVIEW_IMAGE = "config/calibration_preview.jpg"
VIDEO_FOLDER = "videos"

# Canvas stroke colors. All geometry of a lane lives on one canvas, so the color is what tells
# a start line apart from an end line.
POLYGON_COLOR = "#00FF00"
START_LINE_COLOR = "#00AAFF"
END_LINE_COLOR = "#FF0000"
LINE_COLOR_BY_MODE = {"start_line": START_LINE_COLOR, "end_line": END_LINE_COLOR}

DETECTOR_OPTIONS = ["motion", "yolo"]
DEFAULT_CAMERA = {
    "camera_id": "",
    "video_path": "",
    "evidence_root": "evidence",
    "pre_event_seconds": 3.0,
    "post_event_seconds": 1.0,
    "min_contour_area": 900,
    "max_match_distance_px": 75.0,
    "fps_override": None,
    "detector": {"type": "yolo", "yolo_model_path": "yolo11n.pt", "confidence": 0.35, "iou": 0.45, "classes": [2, 3, 5, 7]},
    "lanes": [],
}


def load_preview_image(uploaded_file, video_path: str) -> np.ndarray | None:
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if video_path.strip():
        cap = cv2.VideoCapture(video_path.strip())
        ok, sample = cap.read()
        cap.release()
        if ok:
            return sample

    default_image_path = Path(DEFAULT_PREVIEW_IMAGE)
    if default_image_path.exists():
        image = cv2.imread(str(default_image_path))
        if image is not None:
            return image

    return None


def draw_lane_overlay(image: np.ndarray, lane_state: dict) -> np.ndarray:
    out = image.copy()
    polygon = np.array(lane_state.get("polygon", []), dtype=np.int32)
    if len(polygon) >= 3:
        cv2.polylines(out, [polygon], True, (0, 255, 255), 2)

    measurement = lane_state.get("measurement", {})
    start_lines = measurement.get("start_lines") or ([] if len(measurement.get("start_line", [])) != 2 else [measurement.get("start_line", [])])
    end_lines = measurement.get("end_lines") or ([] if len(measurement.get("end_line", [])) != 2 else [measurement.get("end_line", [])])
    for start_line in start_lines:
        if len(start_line) == 2:
            cv2.line(out, tuple(start_line[0]), tuple(start_line[1]), (0, 255, 0), 3)
    for end_line in end_lines:
        if len(end_line) == 2:
            cv2.line(out, tuple(end_line[0]), tuple(end_line[1]), (0, 0, 255), 3)

    if len(polygon):
        x, y = polygon[0]
        cv2.putText(
            out,
            f"{lane_state['lane_id']} limit={lane_state['speed_limit_kmph']}",
            (int(x), max(20, int(y) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return out


def simplify_polygon(points: list[list[int]]) -> list[list[int]]:
    if len(points) < 3:
        return points
    contour = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    simplified = approx.reshape(-1, 2).tolist()
    return simplified if len(simplified) >= 3 else points


def polygon_canvas_to_points(canvas_data: dict | None) -> list[list[int]]:
    if canvas_data is None:
        return []
    objects = canvas_data.get("objects", [])
    raw_points: list[tuple[int, int]] = []
    for obj in reversed(objects):  # most recently drawn polygon wins
        obj_type = str(obj.get("type", "")).lower()
        if obj_type == "polygon":
            for point in obj.get("points", []):
                if isinstance(point, dict):
                    raw_points.append((int(round(point.get("x", 0))), int(round(point.get("y", 0)))))
                elif isinstance(point, (list, tuple)) and len(point) >= 2:
                    raw_points.append((int(round(point[0])), int(round(point[1]))))
            break
        if obj_type == "path":
            for step in obj.get("path", []):
                if len(step) >= 3 and step[0] in {"M", "L"}:
                    raw_points.append((int(round(step[1])), int(round(step[2]))))
            break
    return simplify_polygon([[x, y] for x, y in raw_points])


def lines_canvas_to_points(canvas_data: dict | None, stroke: str) -> list[list[list[int]]]:
    """Return all line objects drawn with the given stroke color, in drawing order."""
    if canvas_data is None:
        return []

    lines: list[list[list[int]]] = []
    objects = canvas_data.get("objects", [])
    for obj in objects:
        obj_type = str(obj.get("type", "")).lower()
        if obj_type != "line" or str(obj.get("stroke", "")).upper() != stroke.upper():
            continue
        # Fabric/Canvas stores line endpoints relative to the object center.
        # Convert them back to absolute canvas coordinates before saving.
        left = float(obj.get("left", 0))
        top = float(obj.get("top", 0))
        x1 = float(obj.get("x1", 0))
        y1 = float(obj.get("y1", 0))
        x2 = float(obj.get("x2", 0))
        y2 = float(obj.get("y2", 0))

        p1 = (int(round(left + x1)), int(round(top + y1)))
        p2 = (int(round(left + x2)), int(round(top + y2)))
        lines.append([[p1[0], p1[1]], [p2[0], p2[1]]])

    return lines


def blank_lane_state(lane_id: str, name: str, speed_limit_kmph: float = 40.0) -> dict:
    return {
        "lane_id": lane_id,
        "name": name,
        "speed_limit_kmph": speed_limit_kmph,
        "polygon": [],
        "measurement": {
            "start_line": [],
            "end_line": [],
            "start_lines": [],
            "end_lines": [],
            "distance_m": 10.0,
        },
        "canvas_json": None,
    }


def lane_has_geometry(lane_state: dict) -> bool:
    polygon = lane_state.get("polygon") or []
    measurement = lane_state.get("measurement") or {}
    start_line = measurement.get("start_line") or []
    end_line = measurement.get("end_line") or []
    start_lines = measurement.get("start_lines") or []
    end_lines = measurement.get("end_lines") or []
    has_start = len(start_line) == 2 or any(len(line) == 2 for line in start_lines)
    has_end = len(end_line) == 2 or any(len(line) == 2 for line in end_lines)
    return len(polygon) >= 3 or has_start or has_end


def geometry_status(lane_state: dict) -> str:
    measurement = lane_state.get("measurement") or {}
    marks = {
        "polygon": len(lane_state.get("polygon") or []) >= 3,
        "start line": bool(measurement.get("start_lines") or len(measurement.get("start_line") or []) == 2),
        "end line": bool(measurement.get("end_lines") or len(measurement.get("end_line") or []) == 2),
    }
    return " · ".join(f"{name} {'✅' if ok else '❌'}" for name, ok in marks.items())


def build_initial_drawing(lane_state: dict) -> dict | None:
    canvas_json = lane_state.get("canvas_json")
    if canvas_json:
        return canvas_json
    return None


def update_lane_state_from_canvas(lane_state: dict, canvas_json: dict, mode: str) -> bool:
    changed = False

    if mode == "polygon":
        polygon = polygon_canvas_to_points(canvas_json)
        if len(polygon) >= 3 and polygon != lane_state.get("polygon", []):
            lane_state["polygon"] = polygon
            changed = True

    if mode in {"start_line", "end_line"}:
        # Only lines drawn in this mode's color belong to it; the other mode's lines share the canvas.
        lines_points = lines_canvas_to_points(canvas_json, LINE_COLOR_BY_MODE[mode])
        measurement = lane_state.setdefault("measurement", {})
        lines_key = "start_lines" if mode == "start_line" else "end_lines"

        # An empty canvas never erases saved lines (use "Reset lane geometry" for that).
        if lines_points and lines_points != measurement.get(lines_key, []):
            measurement[lines_key] = lines_points
            measurement[mode] = lines_points[0]
            changed = True

    if canvas_json != lane_state.get("canvas_json"):
        lane_state["canvas_json"] = canvas_json
        changed = True

    return changed


def image_to_canvas_background(image: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def load_saved_config(path: str) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def cameras_from_config(saved: dict) -> list[dict]:
    if isinstance(saved.get("cameras"), list) and saved["cameras"]:
        return saved["cameras"]
    if saved.get("lanes"):  # older single-camera file
        return [saved]
    return [{"camera_id": "CAM01"}]


def lane_state_from_config(lane: dict, index: int) -> dict:
    lane_key = f"lane_{index + 1}"
    state = blank_lane_state(
        lane.get("lane_id", lane_key),
        lane.get("name", lane_key.replace("_", " ").title()),
        float(lane.get("speed_limit_kmph", 40.0)),
    )
    state["polygon"] = [list(p) for p in lane.get("polygon", [])]
    measurement = lane.get("measurement", {})
    for mode in ("start_line", "end_line"):
        lines = measurement.get(f"{mode}s") or ([measurement[mode]] if len(measurement.get(mode) or []) == 2 else [])
        state["measurement"][f"{mode}s"] = [[list(p) for p in line] for line in lines]
        state["measurement"][mode] = state["measurement"][f"{mode}s"][0] if lines else []
    state["measurement"]["distance_m"] = float(measurement.get("distance_m", 10.0))
    state["canvas_json"] = lane.get("canvas_json")
    return state


def browse_for_video() -> str | None:
    """Open the operating system's file dialog (the dashboard server runs on this PC)."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        return None
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="Select a video",
        initialdir=str(Path(VIDEO_FOLDER).resolve()) if Path(VIDEO_FOLDER).is_dir() else None,
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")],
    )
    root.destroy()
    return path or None


def to_pretty_json(data: dict) -> str:
    # Keep each point on one line: [x, y] instead of one number per line.
    text = json.dumps(data, indent=2, ensure_ascii=False)
    return re.sub(
        r"\[\s+(-?[\d.]+(?:,\s+-?[\d.]+)*)\s+\]",
        lambda m: "[" + ", ".join(re.split(r",\s+", m.group(1))) + "]",
        text,
    )


# ---- session state: cameras ------------------------------------------------------------------
# Each camera's settings are kept in st.session_state.cams[uid]["values"]. Streamlit deletes the
# state of widgets that are not on screen, so the widgets of the selected camera ("<uid>_<field>"
# keys) are filled from that store before drawing and copied back into it after every run.


def skey(uid: str, field: str) -> str:
    return f"{uid}_{field}"


def init_lane_widgets(uid: str, lane_key: str, lane: dict) -> None:
    defaults = {
        "id": lane["lane_id"],
        "name": lane["name"],
        "limit": float(lane["speed_limit_kmph"]),
        "distance": float(lane["measurement"]["distance_m"]),
    }
    values = st.session_state.cams[uid]["values"]
    for field, value in defaults.items():
        values.setdefault(f"{lane_key}_{field}", value)


def bind_widgets(uid: str) -> None:
    for field, value in st.session_state.cams[uid]["values"].items():
        if skey(uid, field) not in st.session_state:
            st.session_state[skey(uid, field)] = value


def sync_widgets(uid: str) -> None:
    values = st.session_state.cams[uid]["values"]
    for field in values:
        if skey(uid, field) in st.session_state:
            values[field] = st.session_state[skey(uid, field)]


def add_camera(camera: dict) -> str:
    uid = f"cam{st.session_state.next_cam_uid}"
    st.session_state.next_cam_uid += 1
    settings = {**DEFAULT_CAMERA, **camera}
    detector = {**DEFAULT_CAMERA["detector"], **(settings.get("detector") or {})}
    widget_defaults = {
        "camera_id": settings.get("camera_id") or f"CAM{len(st.session_state.cam_order) + 1:02d}",
        "video": settings.get("video_path") or "",
        "evidence_root": settings["evidence_root"],
        "pre": float(settings["pre_event_seconds"]),
        "post": float(settings["post_event_seconds"]),
        "min_area": int(settings["min_contour_area"]),
        "max_match": float(settings["max_match_distance_px"]),
        "fps_enable": settings["fps_override"] is not None,
        "fps": float(settings["fps_override"] or 25.0),
        "det_type": detector["type"] if detector["type"] in DETECTOR_OPTIONS else "yolo",
        "model": detector["yolo_model_path"],
        "conf": float(detector["confidence"]),
        "iou": float(detector["iou"]),
        "classes": ",".join(str(c) for c in detector["classes"]),
        "lane_count": max(1, len(settings["lanes"] or [])),
    }
    st.session_state[skey(uid, "active_lane")] = "lane_1"  # plain state, not a widget

    lanes = {f"lane_{i + 1}": lane_state_from_config(lane, i) for i, lane in enumerate(settings["lanes"] or [])}
    lanes = lanes or {"lane_1": blank_lane_state("lane_1", "Lane 1")}
    st.session_state.cams[uid] = {"lanes": lanes, "canvas_version": 0, "values": widget_defaults}
    for lane_key, lane in lanes.items():
        init_lane_widgets(uid, lane_key, lane)
    st.session_state.cam_order.append(uid)
    return uid


def lane_config(uid: str, lane_key: str) -> dict:
    lane = st.session_state.cams[uid]["lanes"][lane_key]
    values = st.session_state.cams[uid]["values"]
    get = lambda field, default: values.get(f"{lane_key}_{field}", default)
    measurement = lane["measurement"]
    return {
        "lane_id": str(get("id", lane["lane_id"])).strip() or lane_key,
        "name": str(get("name", lane["name"])).strip() or lane_key,
        "speed_limit_kmph": float(get("limit", lane["speed_limit_kmph"])),
        "polygon": lane["polygon"],
        "measurement": {
            "start_line": measurement.get("start_line", []),
            "end_line": measurement.get("end_line", []),
            "start_lines": measurement.get("start_lines", []),
            "end_lines": measurement.get("end_lines", []),
            "distance_m": float(get("distance", measurement.get("distance_m", 10.0))),
        },
    }


def camera_config(uid: str) -> dict:
    cam = st.session_state.cams[uid]
    s = lambda field: cam["values"][field]
    lane_keys = [f"lane_{i + 1}" for i in range(int(s("lane_count")))]
    return {
        "camera_id": str(s("camera_id")).strip(),
        "video_path": str(s("video")).strip(),
        "evidence_root": str(s("evidence_root")).strip() or "evidence",
        "pre_event_seconds": float(s("pre")),
        "post_event_seconds": float(s("post")),
        "min_contour_area": int(s("min_area")),
        "max_match_distance_px": float(s("max_match")),
        "fps_override": float(s("fps")) if s("fps_enable") else None,
        "detector": {
            "type": s("det_type"),
            "yolo_model_path": str(s("model")).strip(),
            "confidence": float(s("conf")),
            "iou": float(s("iou")),
            "classes": [int(v.strip()) for v in str(s("classes")).split(",") if v.strip()],
        },
        "lanes": [
            lane_config(uid, key) for key in lane_keys if key in cam["lanes"] and lane_has_geometry(cam["lanes"][key])
        ],
    }


def on_add_camera() -> None:
    sync_widgets(st.session_state.selected_cam)
    st.session_state.selected_cam = add_camera({})


def on_remove_camera() -> None:
    uid = st.session_state.selected_cam
    if len(st.session_state.cam_order) > 1:
        st.session_state.cam_order.remove(uid)
        st.session_state.cams.pop(uid, None)
        st.session_state.selected_cam = st.session_state.cam_order[0]


def on_browse(uid: str) -> None:
    path = browse_for_video()
    if path:
        st.session_state[skey(uid, "video")] = path
    elif path is None:
        st.session_state.flash = ("info", "No file selected.")


def save_all_cameras() -> None:
    output_path = st.session_state.config_output
    sync_widgets(st.session_state.selected_cam)
    try:
        cameras = [camera_config(uid) for uid in st.session_state.cam_order]
    except ValueError:
        st.session_state.flash = ("error", "YOLO classes must be comma-separated numbers, e.g. 2,3,5,7.")
        return
    names = [c["camera_id"] for c in cameras]
    if any(not n for n in names) or len(set(names)) != len(names):
        st.session_state.flash = ("error", "Every camera needs a unique, non-empty name.")
        return
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(to_pretty_json({"cameras": cameras}) + "\n", encoding="utf-8")
    missing = [c["camera_id"] for c in cameras if not c["lanes"]]
    note = f" (no lanes yet: {', '.join(missing)})" if missing else ""
    st.session_state.flash = ("success", f"Saved {len(cameras)} camera(s) to {out_path}{note}")


st.title("Calibration")
st.caption("Select a camera, then draw lane polygons and start/end lines on its preview image.")

# Start from the saved calibration; it only changes when something is changed and saved.
if "cams" not in st.session_state:
    st.session_state.cams = {}
    st.session_state.cam_order = []
    st.session_state.next_cam_uid = 1
    for saved_camera in cameras_from_config(load_saved_config(DEFAULT_OUTPUT_CONFIG)):
        add_camera(saved_camera)
    st.session_state.selected_cam = st.session_state.cam_order[0]

with st.sidebar:
    st.header("Cameras")
    camera_labels = {u: st.session_state.cams[u]["values"]["camera_id"] or u for u in st.session_state.cam_order}
    uid = st.selectbox("Select camera", options=st.session_state.cam_order, key="selected_cam", format_func=camera_labels.get)
    bind_widgets(uid)
    st.text_input("Camera name", key=skey(uid, "camera_id"), help="Also used as the evidence folder name.")
    col_add, col_remove = st.columns(2)
    col_add.button("Add camera", on_click=on_add_camera, width="stretch")
    col_remove.button(
        "Remove camera", on_click=on_remove_camera, disabled=len(st.session_state.cam_order) <= 1, width="stretch"
    )
    st.divider()
    st.text_input("Config file", value=DEFAULT_OUTPUT_CONFIG, key="config_output")
    st.button("Save calibration config", type="primary", on_click=save_all_cameras, width="stretch")
    st.caption("Saves all cameras. Nothing changes on disk until you press this.")

if "flash" in st.session_state:
    kind, message = st.session_state.pop("flash")
    getattr(st, kind)(message)

cam = st.session_state.cams[uid]
k = lambda field: skey(uid, field)

left, right = st.columns([1.0, 1.2])

with left:
    st.subheader(f"Settings: {st.session_state[k('camera_id')]}")
    st.text_input("Evidence folder", key=k("evidence_root"))

    col1, col2 = st.columns(2)
    with col1:
        st.number_input("Pre-event seconds", min_value=1.0, step=1.0, key=k("pre"))
        st.number_input("Min contour area", min_value=100, step=100, key=k("min_area"))
    with col2:
        st.number_input("Post-event seconds", min_value=1.0, step=1.0, key=k("post"))
        st.number_input("Max track match distance (px)", min_value=10.0, key=k("max_match"))

    st.checkbox("Use FPS override", key=k("fps_enable"))
    if st.session_state[k("fps_enable")]:
        st.number_input("FPS override", min_value=1.0, step=1.0, key=k("fps"))

    st.subheader("Detector")
    st.selectbox("Detector type", options=DETECTOR_OPTIONS, key=k("det_type"))
    st.text_input("YOLO model path", key=k("model"))
    st.slider("YOLO confidence", min_value=0.05, max_value=0.95, step=0.05, key=k("conf"))
    st.slider("YOLO IoU", min_value=0.10, max_value=0.95, step=0.05, key=k("iou"))
    st.text_input("YOLO classes (comma-separated COCO IDs)", key=k("classes"))

    st.subheader("Lanes")
    lane_count = int(st.number_input("Number of lanes", min_value=1, step=1, key=k("lane_count")))
    lane_options = [f"lane_{index + 1}" for index in range(lane_count)]
    for lane_key in lane_options:
        if lane_key not in cam["lanes"]:
            cam["lanes"][lane_key] = blank_lane_state(lane_key, lane_key.replace("_", " ").title())
        init_lane_widgets(uid, lane_key, cam["lanes"][lane_key])
    bind_widgets(uid)  # fields of newly added lanes
    if st.session_state[k("active_lane")] not in lane_options:
        st.session_state[k("active_lane")] = lane_options[0]

    for lane_key in lane_options:
        lane_state = cam["lanes"][lane_key]
        is_active = lane_key == st.session_state[k("active_lane")]
        title = f"{st.session_state[k(f'{lane_key}_name')]}{'  (editing)' if is_active else ''}"
        with st.expander(title, expanded=is_active):
            if not is_active and st.button("Edit this lane", key=k(f"{lane_key}_edit")):
                st.session_state[k("active_lane")] = lane_key
                st.rerun()
            st.text_input("Lane ID", key=k(f"{lane_key}_id"))
            st.text_input("Lane name", key=k(f"{lane_key}_name"))
            st.number_input("Speed limit (km/h)", min_value=1.0, step=1.0, key=k(f"{lane_key}_limit"))
            st.number_input("Distance between lines (m)", min_value=0.1, step=0.5, key=k(f"{lane_key}_distance"))
            st.caption(geometry_status(lane_state))
            if st.button("Reset lane geometry", key=k(f"{lane_key}_reset")):
                cam["lanes"][lane_key] = blank_lane_state(lane_key, lane_state["name"], float(lane_state["speed_limit_kmph"]))
                cam["canvas_version"] += 1
                st.rerun()

with right:
    st.subheader("Preview and Drawing")
    col_video, col_browse = st.columns([4, 1], vertical_alignment="bottom")
    col_video.text_input(
        "Video file or live stream (its first frame is the preview)",
        key=k("video"),
        help="A video file, or a live camera address such as rtsp://user:password@192.168.1.64:554/stream1",
    )
    col_browse.button("Browse...", on_click=on_browse, args=(uid,), key=k("browse"), width="stretch")
    uploaded_image = st.file_uploader("Or upload a sample frame image", type=["jpg", "jpeg", "png"], key=k("upload"))

    frame = load_preview_image(uploaded_image, st.session_state[k("video")])
    active_lane = st.session_state[k("active_lane")]
    active_lane_state = cam["lanes"][active_lane]

    if frame is not None:
        preview_frame = draw_lane_overlay(frame.copy(), lane_config(uid, active_lane))
        preview_rgb = cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB)
        height, width = preview_rgb.shape[:2]

        st.image(preview_rgb, caption="Calibration preview", width="stretch")
        st.markdown(f"#### Editing: {st.session_state[k(f'{active_lane}_name')]}")

        mode = st.radio(
            "Select geometry to draw", options=["polygon", "start_line", "end_line"], horizontal=True, key=k("mode")
        )

        # Clears only the drawing; the lane's saved geometry stays until a new drawing is saved.
        if st.button("Clear current canvas", key=k(f"clear_canvas_{active_lane}")):
            active_lane_state["canvas_json"] = None
            cam["canvas_version"] += 1
            st.rerun()

        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.08)",
            stroke_width=3,
            stroke_color=POLYGON_COLOR if mode == "polygon" else LINE_COLOR_BY_MODE[mode],
            background_image=image_to_canvas_background(frame),
            update_streamlit=True,
            height=height,
            width=width,
            drawing_mode="polygon" if mode == "polygon" else "line",
            initial_drawing=build_initial_drawing(active_lane_state),
            key=f"canvas_{uid}_{active_lane}_{cam['canvas_version']}",
        )

        # Drawings are applied to the lane only with this button, so opening the dashboard or
        # switching modes never changes the loaded calibration.
        if st.button("Save current drawing to active lane"):
            if canvas_result.json_data is None:
                st.warning("Draw on the canvas first.")
            else:
                updated = update_lane_state_from_canvas(active_lane_state, canvas_result.json_data, mode)
                measurement = active_lane_state["measurement"]
                lines_key = "start_lines" if mode == "start_line" else "end_lines"
                if mode == "polygon" and len(active_lane_state.get("polygon", [])) < 3:
                    st.warning("Draw a closed polygon on the canvas first.")
                elif mode in {"start_line", "end_line"} and not (
                    len(measurement.get(mode, [])) == 2 or any(len(line) == 2 for line in measurement.get(lines_key, []))
                ):
                    st.warning("Draw a complete line on the canvas first.")
                else:
                    st.session_state.flash = (
                        "success",
                        f"Saved {mode.replace('_', ' ')} for {active_lane}{'' if updated else ' (no changes detected)'}. "
                        "Press 'Save calibration config' to write it to disk.",
                    )
                    st.rerun()

        st.markdown("---")
        st.subheader("All lanes of this camera")
        saved_preview = frame.copy()
        for lane_key in lane_options:
            if lane_has_geometry(cam["lanes"][lane_key]):
                saved_preview = draw_lane_overlay(saved_preview, lane_config(uid, lane_key))
        st.image(cv2.cvtColor(saved_preview, cv2.COLOR_BGR2RGB), caption="All lanes", width="stretch")
    else:
        st.info("Choose this camera's video (Browse...) or upload a frame image to start calibration.")

sync_widgets(uid)

st.markdown("---")
st.caption("Tip: pick one lane at a time, draw polygon/start/end on the preview, save the drawing, then save the config.")
