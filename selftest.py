"""Checks that this version of the program works on this computer. Run: python selftest.py

Used by update.py after installing a new version (a version that fails is removed again), and useful on its
own after setting up a computer. It needs no camera and no video: it makes a short video of a box moving at a
known speed, measures it with the real pipeline and compares the result.
"""

from __future__ import annotations

import csv
import json
import py_compile
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config" / "calibration.json"

# The test video: a box moving down the picture at 6 pixels per frame, 25 frames per second, across two
# lines 150 pixels apart that are declared to be 10 m apart: 1 second for 10 m = 36 km/h.
WIDTH, HEIGHT, FPS = 640, 360, 25.0
BOX_SPEED_PX, LINE_1_Y, LINE_2_Y, DISTANCE_M = 6, 120, 270, 10.0
EXPECTED_KMPH = DISTANCE_M / ((LINE_2_Y - LINE_1_Y) / BOX_SPEED_PX / FPS) * 3.6
TOLERANCE = 0.05


def check_files_compile() -> None:
    for path in [*ROOT.glob("*.py"), *ROOT.glob("src/*.py"), *ROOT.glob("app/*.py")]:
        py_compile.compile(str(path), doraise=True)


def check_calibration_loads() -> None:
    """The cameras already set up on this computer must still be readable by this version."""
    if not CONFIG.exists():
        return
    from src.config import list_camera_ids, load_config

    for camera_id in list_camera_ids(CONFIG):
        load_config(CONFIG, camera_id)


def check_measured_speed(folder: Path) -> None:
    import cv2
    import numpy as np

    from src.pipeline import run

    video = folder / "moving_box.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (WIDTH, HEIGHT))
    for index in range(110):
        frame = np.full((HEIGHT, WIDTH, 3), 60, np.uint8)
        top = (index - 30) * BOX_SPEED_PX  # an empty road first, for the motion detector to learn
        if index >= 30:
            cv2.rectangle(frame, (280, top), (360, top + 60), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()

    config = folder / "calibration.json"
    camera = {
        "camera_id": "SELFTEST",
        "evidence_root": str(folder / "evidence"),
        "pre_event_seconds": 1.0,
        "post_event_seconds": 1.0,
        "detector": {"type": "motion"},
        "lanes": [
            {
                "lane_id": "lane_1",
                "speed_limit_kmph": 10.0,  # the box is "speeding", so an evidence clip is written too
                "polygon": [[0, 0], [WIDTH - 1, 0], [WIDTH - 1, HEIGHT - 1], [0, HEIGHT - 1]],
                "measurement": {
                    "start_line": [[0, LINE_1_Y], [WIDTH - 1, LINE_1_Y]],
                    "end_line": [[0, LINE_2_Y], [WIDTH - 1, LINE_2_Y]],
                    "distance_m": DISTANCE_M,
                },
            }
        ],
    }
    config.write_text(json.dumps({"cameras": [camera]}), encoding="utf-8")

    run(str(video), str(config))

    with (folder / "evidence" / "speed_events.csv").open(newline="", encoding="utf-8") as f:
        events = list(csv.DictReader(f))
    if len(events) != 1:
        raise AssertionError(f"expected 1 measured vehicle in the test video, got {len(events)}")
    speed = float(events[0]["speed_kmph"])
    if abs(speed - EXPECTED_KMPH) > TOLERANCE * EXPECTED_KMPH:
        raise AssertionError(f"the test video's {EXPECTED_KMPH:.1f} km/h was measured as {speed:.1f} km/h")
    if not events[0]["clip_path"] or not Path(events[0]["clip_path"]).exists():
        raise AssertionError("no evidence clip was saved for the test video")


def check_vehicle_detector() -> None:
    import numpy as np

    from src.detector import YoloVehicleDetector

    detector = YoloVehicleDetector(model_path=str(ROOT / "yolo11n.pt"))  # also prepares the OpenVINO copy
    detector.detect(np.zeros((720, 1280, 3), np.uint8))
    print(f"self-test: vehicle detector runs on: {detector.engine}")


def main() -> int:
    sys.path.insert(0, str(ROOT))
    with tempfile.TemporaryDirectory() as folder:
        checks = [
            ("program files", check_files_compile),
            ("this computer's calibration", check_calibration_loads),
            ("speed measurement", lambda: check_measured_speed(Path(folder))),
            ("vehicle detector", check_vehicle_detector),
        ]
        for name, check in checks:
            try:
                check()
            except Exception as exc:  # any failure means this version must not be used
                print(f"SELF-TEST FAILED at {name}: {type(exc).__name__}: {exc}")
                return 1
            print(f"self-test: {name} OK")
    print("SELF-TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
