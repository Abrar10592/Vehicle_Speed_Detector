from __future__ import annotations

import json
import math
import re
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas  # type: ignore[import-not-found]
from PIL import Image

from src.video_source import STREAM_PATHS, camera_login, camera_stream_address, grab_stream_frame, is_live_source


DEFAULT_OUTPUT_CONFIG = "config/calibration.json"
DEFAULT_PREVIEW_IMAGE = "config/calibration_preview.jpg"
VIDEO_FOLDER = "videos"

# Canvas stroke colors. All geometry of a lane lives on one canvas, so the color is what tells
# a start line apart from an end line.
POLYGON_COLOR = "#00FF00"
START_LINE_COLOR = "#00AAFF"
END_LINE_COLOR = "#FF0000"
LINE_COLOR_BY_MODE = {"start_line": START_LINE_COLOR, "end_line": END_LINE_COLOR}
MODE_LABELS = {"polygon": "Lane area (polygon)", "start_line": "Start line", "end_line": "End line"}
MODE_HELP = {
    "polygon": "Click each corner of the lane area, then **click the first corner again to close it**. "
    "A polygon that is not closed is not kept.",
    "start_line": "Press at one end of the **start line** (blue) and drag to the other end. "
    "Several pieces are allowed; together they form the line.",
    "end_line": "Press at one end of the **end line** (red) and drag to the other end. "
    "Several pieces are allowed; together they form the line.",
}
# Width options for the drawing; the frame is scaled to fit and coordinates are saved at full video resolution.
CANVAS_WIDTHS = [640, 800, 960, 1120, 1280, 1600, 1920]
# Zoomed placement of line ends: magnifications, and the size of the magnified picture on screen.
ZOOM_LEVELS = [2, 4, 6, 10]
ZOOM_VIEW_SIZE = (840, 480)
LINE_JOIN_PX = 12  # ends of two pieces of one line this close together are one corner and move together

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


def live_camera_frame(address: str, refresh: bool = False) -> np.ndarray | None:
    """A live camera's picture is fetched once and kept: the page reruns on every click, and reconnecting to the
    camera each time would make drawing slow and change the picture under the drawing."""
    frames = st.session_state.setdefault("live_frames", {})
    if refresh or address not in frames:
        frames[address] = grab_stream_frame(address)
    return frames[address]


def load_preview_image(uploaded_file, video_path: str) -> np.ndarray | None:
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if is_live_source(video_path.strip()):
        frame = live_camera_frame(video_path.strip())
        if frame is not None:
            return frame
    elif video_path.strip():
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


def hex_to_bgr(color: str) -> tuple[int, int, int]:
    r, g, b = (int(color[i : i + 2], 16) for i in (1, 3, 5))
    return b, g, r


def draw_lane_overlay(image: np.ndarray, lane_state: dict) -> np.ndarray:
    """The lane's saved geometry, in the same colors as the drawing tools."""
    out = image.copy()
    unit = max(1.0, image.shape[1] / 1280)  # keeps marks visible on large frames shown scaled down
    polygon = np.array(lane_state.get("polygon", []), dtype=np.int32)
    if len(polygon) >= 3:
        cv2.polylines(out, [polygon], True, hex_to_bgr(POLYGON_COLOR), round(2 * unit))

    measurement = lane_state.get("measurement", {})
    for mode, color in LINE_COLOR_BY_MODE.items():
        lines = measurement.get(f"{mode}s") or ([measurement[mode]] if len(measurement.get(mode) or []) == 2 else [])
        for line in lines:
            if len(line) == 2:
                cv2.line(out, tuple(line[0]), tuple(line[1]), hex_to_bgr(color), round(3 * unit))

    if len(polygon):
        x, y = polygon[0]
        cv2.putText(
            out,
            f"{lane_state['lane_id']} limit={lane_state['speed_limit_kmph']}",
            (int(x), max(round(20 * unit), int(y) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6 * unit,
            (255, 255, 255),
            round(2 * unit),
            cv2.LINE_AA,
        )
    return out


def drop_repeated_points(points: list[list[int]]) -> list[list[int]]:
    """Remove accidental double clicks; every corner the user placed is kept."""
    kept: list[list[int]] = []
    for point in points:
        if not kept or point != kept[-1]:
            kept.append(point)
    if len(kept) > 1 and kept[0] == kept[-1]:
        kept.pop()  # closing click on the first corner
    return kept


def placed(obj: dict, dx: float, dy: float) -> tuple[float, float]:
    """Canvas position of a point given as an offset from the object's center.

    Fabric keeps a shape's own points as drawn and records moves, resizes and rotations made with the toolbar's
    Edit button separately (center left/top, scaleX/Y, angle), so they are applied here.
    """
    sx, sy = float(obj.get("scaleX") or 1), float(obj.get("scaleY") or 1)
    angle = math.radians(float(obj.get("angle") or 0))
    dx, dy = dx * sx, dy * sy
    return (
        float(obj.get("left", 0)) + dx * math.cos(angle) - dy * math.sin(angle),
        float(obj.get("top", 0)) + dx * math.sin(angle) + dy * math.cos(angle),
    )


def polygon_canvas_to_points(canvas_data: dict | None, to_image) -> list[list[int]]:
    """The most recently drawn closed polygon, in image coordinates (`to_image` maps canvas -> image)."""
    if canvas_data is None:
        return []
    objects = canvas_data.get("objects", [])
    raw_points: list[list[int]] = []
    for obj in reversed(objects):  # most recently drawn polygon wins
        obj_type = str(obj.get("type", "")).lower()
        if obj_type == "polygon":
            points = [(p["x"], p["y"]) if isinstance(p, dict) else (p[0], p[1]) for p in obj.get("points", [])]
            if points:
                xs, ys = [x for x, _ in points], [y for _, y in points]
                cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2  # the polygon's center as drawn
                raw_points = [to_image(*placed(obj, x - cx, y - cy)) for x, y in points]
            break
        if obj_type == "path":
            for step in obj.get("path", []):
                if len(step) >= 3 and step[0] in {"M", "L"}:
                    raw_points.append(to_image(step[1], step[2]))
            break
    return drop_repeated_points(raw_points)


def lines_canvas_to_points(canvas_data: dict | None, stroke: str, to_image) -> list[list[list[int]]]:
    """All line objects drawn with the given stroke color, in drawing order, in image coordinates."""
    if canvas_data is None:
        return []

    lines: list[list[list[int]]] = []
    for obj in canvas_data.get("objects", []):
        obj_type = str(obj.get("type", "")).lower()
        if obj_type != "line" or str(obj.get("stroke", "")).upper() != stroke.upper():
            continue
        # Fabric stores line endpoints as offsets from the line's center.
        p1 = to_image(*placed(obj, float(obj.get("x1", 0)), float(obj.get("y1", 0))))
        p2 = to_image(*placed(obj, float(obj.get("x2", 0)), float(obj.get("y2", 0))))
        if p1 != p2:  # a click without a drag
            lines.append([p1, p2])

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


def apply_drawing(lane_state: dict, canvas_json: dict | None, mode: str, to_image) -> str | None:
    """Store the canvas drawing for `mode` in the lane. Returns what is missing from the drawing, or None."""
    if mode == "polygon":
        polygon = polygon_canvas_to_points(canvas_json, to_image)
        if len(polygon) < 3:
            return "No closed polygon on the canvas yet. Click each corner, then click the first corner again to close it."
        lane_state["polygon"] = polygon
        return None

    # Only lines drawn in this mode's color belong to it; the other mode's lines share the canvas.
    lines = lines_canvas_to_points(canvas_json, LINE_COLOR_BY_MODE[mode], to_image)
    if not lines:
        return f"No {MODE_LABELS[mode].lower()} on the canvas yet. Press at one end of the line and drag to the other."
    measurement = lane_state.setdefault("measurement", {})
    measurement[f"{mode}s"] = lines
    measurement[mode] = lines[0]
    return None


def line_ends(lane_state: dict) -> list[tuple[str, int, int]]:
    """Every end of the lane's start and end lines, as (mode, piece number, 0 or 1)."""
    measurement = lane_state.get("measurement") or {}
    return [
        (mode, piece, end)
        for mode in LINE_COLOR_BY_MODE
        for piece, line in enumerate(measurement.get(f"{mode}s") or [])
        if len(line) == 2
        for end in range(2)
    ]


def line_end_label(lane_state: dict, line_end: tuple[str, int, int]) -> str:
    mode, piece, end = line_end
    lines = lane_state["measurement"][f"{mode}s"]
    side = "left" if lines[piece][end][0] <= lines[piece][1 - end][0] else "right"
    piece_text = f", piece {piece + 1}" if len(lines) > 1 else ""
    return f"{MODE_LABELS[mode]}{piece_text}: {side} end"


def move_line_end(lane_state: dict, line_end: tuple[str, int, int], point: list[int]) -> None:
    """Move one end of a line. Ends of the line's other pieces that were drawn onto it move with it."""
    mode, piece, end = line_end
    lines = lane_state["measurement"][f"{mode}s"]
    old = tuple(lines[piece][end])
    for line in lines:
        for k in range(2):
            if math.dist(line[k], old) <= LINE_JOIN_PX:
                line[k] = list(point)
    lines[piece][end] = list(point)
    lane_state["measurement"][mode] = lines[0]


def zoomed_view(frame: np.ndarray, lane_state: dict, line_end: tuple[str, int, int], zoom: int) -> tuple[np.ndarray, int, int]:
    """The frame around a line end, magnified `zoom` times, with the lane's lines and a cross on that end.

    Returns the picture and the frame position of its top left corner.
    """
    mode, piece, end = line_end
    measurement = lane_state["measurement"]
    cx, cy = measurement[f"{mode}s"][piece][end]
    height, width = frame.shape[:2]
    crop_w, crop_h = min(ZOOM_VIEW_SIZE[0] // zoom, width), min(ZOOM_VIEW_SIZE[1] // zoom, height)
    x0 = min(max(cx - crop_w // 2, 0), width - crop_w)
    y0 = min(max(cy - crop_h // 2, 0), height - crop_h)
    view = cv2.resize(
        frame[y0 : y0 + crop_h, x0 : x0 + crop_w], (crop_w * zoom, crop_h * zoom), interpolation=cv2.INTER_CUBIC
    )

    def to_view(point: list[int]) -> tuple[int, int]:  # the middle of that frame pixel
        return round((point[0] - x0 + 0.5) * zoom), round((point[1] - y0 + 0.5) * zoom)

    for line_mode, color in LINE_COLOR_BY_MODE.items():
        for line in measurement.get(f"{line_mode}s") or []:
            if len(line) == 2:
                cv2.line(view, to_view(line[0]), to_view(line[1]), hex_to_bgr(color), 1, cv2.LINE_AA)
    # A cross with an open middle, so the spot it points at stays visible.
    vx, vy = to_view([cx, cy])
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        cv2.line(view, (vx + dx * 8, vy + dy * 8), (vx + dx * 26, vy + dy * 26), (0, 255, 255), 2, cv2.LINE_AA)
    return view, x0, y0


def clicked_point(canvas_data: dict | None) -> tuple[float, float] | None:
    """Where the canvas was last clicked in point mode (the click is the middle of the dot it leaves)."""
    for obj in reversed((canvas_data or {}).get("objects", [])):
        if str(obj.get("type", "")).lower() == "circle":
            half = float(obj.get("radius") or 0) + float(obj.get("strokeWidth") or 0) / 2
            x = float(obj.get("left", 0)) + {"left": half, "right": -half}.get(obj.get("originX"), 0.0)
            y = float(obj.get("top", 0)) + {"top": half, "bottom": -half}.get(obj.get("originY"), 0.0)
            return x, y
    return None


def image_to_canvas_background(image: np.ndarray, width: int, height: int) -> Image.Image:
    small = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))


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
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.mts *.m2ts"), ("All files", "*.*")],
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


@st.dialog("Connect a live camera")
def live_camera_dialog(uid: str) -> None:
    values = st.session_state.cams[uid]["values"]
    saved = camera_login(str(st.session_state.get(skey(uid, "video"), values["video"])))
    brands = [*STREAM_PATHS, "Other"]
    saved_brand = next((b for b, path in STREAM_PATHS.items() if path == saved["path"]), "Other" if saved["path"] else brands[0])
    brand = st.selectbox("Camera brand", brands, index=brands.index(saved_brand))

    with st.form("live_camera", border=False):
        name = st.text_input("Camera name", value=st.session_state.get(skey(uid, "camera_id"), values["camera_id"]))
        ip = st.text_input("Camera IP address", value=saved["ip"], placeholder="192.168.1.101")
        user = st.text_input("User name", value=saved["user"] or "admin")
        password = st.text_input("Password", value=saved["password"], type="password")
        path = STREAM_PATHS.get(brand) or st.text_input(
            "Stream path", value=saved["path"], placeholder="/stream1", help="The part of the stream address after the port; see the camera's manual."
        )
        connect = st.form_submit_button("Connect", type="primary")

    if not connect:
        return
    ip, name = ip.strip(), name.strip()
    if not ip or not name or not path.strip():
        st.error("Fill in the camera name and IP address" + ("." if brand in STREAM_PATHS else ", and the stream path."))
        return
    address = camera_stream_address(ip, user.strip(), password, path.strip())
    with st.spinner("Connecting to the camera..."):
        frame = grab_stream_frame(address)
    if frame is None:
        st.error(
            "No picture from the camera. Check the IP address, user name and password, "
            "and that the camera is switched on and connected."
        )
        return
    st.session_state.setdefault("live_frames", {})[address] = frame
    # The camera's name and video boxes are already on the page; they are filled in at the start of the next run.
    st.session_state.connected_camera = (uid, name, address)
    st.rerun()


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

if "connected_camera" in st.session_state:  # from the live camera dialog
    target, name, address = st.session_state.pop("connected_camera")
    st.session_state.cams[target]["values"].update(camera_id=name, video=address)
    st.session_state[skey(target, "camera_id")] = name
    st.session_state[skey(target, "video")] = address
    st.session_state.flash = ("success", f"Connected to {name}. Its picture is below; draw the lanes on it.")

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
    st.subheader("Video")
    col_video, col_browse, col_live = st.columns([3, 1, 1.4], vertical_alignment="bottom")
    col_video.text_input(
        "Video file or live stream (its first frame is the preview)",
        key=k("video"),
        help="A video file (Browse...), or a live camera's address (Live camera... fills it in).",
    )
    col_browse.button("Browse...", on_click=on_browse, args=(uid,), key=k("browse"), width="stretch")
    if col_live.button("Live camera...", key=k("live"), width="stretch"):
        live_camera_dialog(uid)
    uploaded_image = st.file_uploader("Or upload a sample frame image", type=["jpg", "jpeg", "png"], key=k("upload"))

    video = st.session_state[k("video")].strip()
    if is_live_source(video):
        if st.button("Refresh camera picture", icon=":material/refresh:", key=k("refresh_live")):
            live_camera_frame(video, refresh=True)
        if uploaded_image is None and live_camera_frame(video) is None:
            st.warning("No picture from this camera. Check its details with Live camera..., then refresh.")
    frame = load_preview_image(uploaded_image, video)
    if frame is not None:
        all_lanes = frame.copy()
        for lane_key in lane_options:
            if lane_has_geometry(cam["lanes"][lane_key]):
                all_lanes = draw_lane_overlay(all_lanes, lane_config(uid, lane_key))
        st.image(cv2.cvtColor(all_lanes, cv2.COLOR_BGR2RGB), caption="All lanes of this camera", width="stretch")
    else:
        st.info("Choose this camera's video (Browse...) or upload a frame image to start calibration.")

# The drawing gets the full page width, so large frames fit without sideways scrolling.
if frame is not None:
    active_lane = st.session_state[k("active_lane")]
    active_lane_state = cam["lanes"][active_lane]
    height, width = frame.shape[:2]

    st.divider()
    st.subheader(f"Draw: {st.session_state[k(f'{active_lane}_name')]}")
    chosen_lane = st.segmented_control(
        "Lane",
        lane_options,
        format_func=lambda key: st.session_state[k(f"{key}_name")],
        default=active_lane,
        key=k(f"draw_lane_{active_lane}"),
    )
    if chosen_lane and chosen_lane != active_lane:
        st.session_state[k("active_lane")] = chosen_lane
        st.rerun()

    st.session_state.setdefault("canvas_width", 800)  # fits a laptop screen; larger sizes for bigger monitors
    with st.container(horizontal=True, vertical_alignment="bottom", gap="large"):
        mode = st.radio("What to draw", options=list(MODE_LABELS), format_func=MODE_LABELS.get, horizontal=True, key=k("mode"))
        st.select_slider(
            "Drawing size",
            options=CANVAS_WIDTHS,
            format_func=lambda w: f"{w} px",
            key="canvas_width",
            width=320,
            help="Pick the largest size that fits on your screen without scrolling sideways.",
        )
        # Clears only the drawing; the lane's saved geometry stays until a new drawing is saved.
        if st.button("Clear drawing", icon=":material/ink_eraser:", key=k(f"clear_canvas_{active_lane}")):
            cam["canvas_version"] += 1
            st.rerun()
    st.caption(
        MODE_HELP[mode] + " The lane's saved geometry is shown on the picture. The small toolbar above the "
        "picture's top right corner has Undo, Redo, Edit (move or resize a shape) and Reset."
    )

    canvas_width = min(st.session_state.canvas_width, width)
    scale = width / canvas_width
    canvas_height = round(height / scale)

    def to_image(x: float, y: float) -> list[int]:
        """Canvas pixel -> full-resolution frame pixel."""
        return [min(max(round(float(x) * scale), 0), width - 1), min(max(round(float(y) * scale), 0), height - 1)]

    canvas_result = st_canvas(
        fill_color="rgba(0, 255, 0, 0.12)",
        stroke_width=3,
        stroke_color=POLYGON_COLOR if mode == "polygon" else LINE_COLOR_BY_MODE[mode],
        background_image=image_to_canvas_background(
            draw_lane_overlay(frame, lane_config(uid, active_lane)), canvas_width, canvas_height
        ),
        update_streamlit=True,
        height=canvas_height,
        width=canvas_width,
        drawing_mode="polygon" if mode == "polygon" else "line",
        key=f"canvas_{uid}_{active_lane}_{cam['canvas_version']}_{canvas_width}",
    )

    # Drawings are applied to the lane only with this button, so opening the dashboard or
    # switching modes never changes the loaded calibration.
    if st.button(f"Save {MODE_LABELS[mode].lower()} to {st.session_state[k(f'{active_lane}_name')]}", type="primary"):
        problem = apply_drawing(active_lane_state, canvas_result.json_data, mode, to_image)
        if problem:
            st.warning(problem)
        else:
            st.session_state.flash = (
                "toast",
                f"Saved {MODE_LABELS[mode].lower()} for {active_lane}. "
                "Press 'Save calibration config' to write it to disk.",
            )
            st.rerun()

    # The drawing above is scaled down, so a line drawn on it can be several frame pixels off, and far from the
    # camera one pixel is a long stretch of road. Each line end is therefore placed again on a magnified picture.
    ends = line_ends(active_lane_state)
    if ends:
        st.divider()
        st.subheader("Place the line ends exactly")
        st.caption(
            "Speeds are only as accurate as the lines. Choose a line end, then click in the magnified picture "
            "exactly where the line must touch the road, e.g. where your cone or mark meets the ground. "
            "The yellow cross shows where that end is now; the picture follows it, so click towards a mark that "
            "is out of view (or use 2x) to reach it. Do this for every end of both lines."
        )
        if st.session_state.get(k(f"zoom_end_{active_lane}")) not in ends:
            st.session_state.pop(k(f"zoom_end_{active_lane}"), None)  # that line was redrawn with fewer pieces
        st.session_state.setdefault(k("zoom"), 4)
        with st.container(horizontal=True, vertical_alignment="bottom", gap="large"):
            line_end = st.selectbox(
                "Line end",
                ends,
                format_func=lambda e: line_end_label(active_lane_state, e),
                key=k(f"zoom_end_{active_lane}"),
                width=360,
            )
            zoom = st.radio("Zoom", ZOOM_LEVELS, format_func=lambda z: f"{z}x", horizontal=True, key=k("zoom"))

        view, view_x0, view_y0 = zoomed_view(frame, active_lane_state, line_end, zoom)
        end_mode, end_piece, end_index = line_end
        zoom_result = st_canvas(
            fill_color="#FFFF00",
            stroke_width=1,
            stroke_color="#FFFF00",
            background_image=Image.fromarray(cv2.cvtColor(view, cv2.COLOR_BGR2RGB)),
            update_streamlit=True,
            height=view.shape[0],
            width=view.shape[1],
            drawing_mode="point",
            point_display_radius=2,
            # A new canvas after every placement, so each click is used once.
            key=f"zoom_{uid}_{active_lane}_{end_mode}_{end_piece}_{end_index}_{zoom}_{cam.setdefault('zoom_version', 0)}",
        )
        end_x, end_y = active_lane_state["measurement"][f"{end_mode}s"][end_piece][end_index]
        st.caption(f"This end is at pixel ({end_x}, {end_y}) of the {width} x {height} picture.")

        click = clicked_point(zoom_result.json_data)
        if click is not None:
            move_line_end(
                active_lane_state,
                line_end,
                [
                    min(max(view_x0 + int(click[0] // zoom), 0), width - 1),
                    min(max(view_y0 + int(click[1] // zoom), 0), height - 1),
                ],
            )
            cam["zoom_version"] += 1
            st.session_state.flash = ("toast", "Line end moved. Press 'Save calibration config' to write it to disk.")
            st.rerun()

sync_widgets(uid)

st.markdown("---")
st.caption("Tip: pick one lane at a time, draw its area, start line and end line, save each drawing, then save the config.")
