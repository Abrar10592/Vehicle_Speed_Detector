from __future__ import annotations

from typing import List

import cv2
import numpy as np

from .geometry import road_point
from .models import Detection


class MotionVehicleDetector:
    def __init__(self, min_contour_area: int = 900) -> None:
        self.min_contour_area = min_contour_area
        self.subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)

    def detect(self, frame: np.ndarray, roi_mask: np.ndarray | None = None) -> List[Detection]:
        fg_mask = self.subtractor.apply(frame)
        _, binary = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel, iterations=2)

        if roi_mask is not None:
            binary = cv2.bitwise_and(binary, roi_mask)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections: list[Detection] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_contour_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            detections.append(Detection(bbox=(x, y, x + w, y + h), score=1.0))

        return detections


Area = tuple[int, int, int, int]  # x1, y1, x2, y2 in the frame

MAIN_IMAGE_SIZE = 640
FAR_IMAGE_SIZE = 960


def _clamped(x1: float, y1: float, x2: float, y2: float, frame_shape: tuple[int, ...]) -> Area:
    height, width = frame_shape[:2]
    return max(0, int(x1)), max(0, int(y1)), min(width, int(x2)), min(height, int(y2))


def _line_box(lines: list) -> tuple[Area, float]:
    """The box around a line's pieces and its diagonal, which is about the line's length."""
    xs = [p[0] for line in lines for p in line]
    ys = [p[1] for line in lines for p in line]
    return (min(xs), min(ys), max(xs), max(ys)), float(np.hypot(max(xs) - min(xs), max(ys) - min(ys)))


def detection_areas(frame_shape: tuple[int, ...], lanes: list) -> tuple[Area, list[Area]]:
    """Where the detector looks: the area around the lanes, and a closer look at each lane's far line.

    The detector shrinks whatever picture it is given to a fixed size, so on a large frame distant vehicles
    become a few pixels and are missed: a motorcycle 28 m from a 4-megapixel camera was only found after it had
    passed the far line. Searching the lane area instead of the whole frame wastes none of that size on sky and
    trees, and the surroundings of a line that is much shorter in the picture than the other one (so much
    farther away) are searched a second time at full resolution.
    """
    xs = [p[0] for lane in lanes for p in lane.polygon]
    ys = [p[1] for lane in lanes for p in lane.polygon]
    lane_w, lane_h = max(xs) - min(xs), max(ys) - min(ys)
    # Vehicles stand above the road they drive on, hence the room above the lanes.
    main = _clamped(
        min(xs) - 0.08 * lane_w, min(ys) - 0.5 * lane_h, max(xs) + 0.08 * lane_w, max(ys) + 0.05 * lane_h, frame_shape
    )
    main_shrink = MAIN_IMAGE_SIZE / max(main[2] - main[0], main[3] - main[1])

    far_areas: list[Area] = []
    for lane in lanes:
        boxes = sorted((_line_box(lane.measurement.start_lines), _line_box(lane.measurement.end_lines)), key=lambda b: b[1])
        (far_box, far_len), (near_box, near_len) = boxes
        if far_len >= 0.6 * near_len:
            continue  # both lines are about as far away
        # From half a line length around the far line to halfway towards the near line.
        middle_x = (far_box[0] + far_box[2] + near_box[0] + near_box[2]) / 4
        middle_y = (far_box[1] + far_box[3] + near_box[1] + near_box[3]) / 4
        pad = 0.5 * far_len
        area = _clamped(
            min(far_box[0] - pad, middle_x),
            min(far_box[1] - pad, middle_y),
            max(far_box[2] + pad, middle_x),
            max(far_box[3] + pad, middle_y),
            frame_shape,
        )
        far_shrink = min(1.0, FAR_IMAGE_SIZE / max(area[2] - area[0], area[3] - area[1]))
        if far_shrink >= 1.5 * main_shrink:  # otherwise the second look shows nothing the first one does not
            far_areas.append(area)
    return main, far_areas


def _overlap_of_smaller(a: Area, b: Area) -> float:
    """The part of the smaller box that lies inside the other one (1 = entirely)."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return w * h / smaller if smaller > 0 else 0.0


class YoloVehicleDetector:
    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        confidence: float = 0.35,
        iou: float = 0.45,
        classes: list[int] | None = None,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as ex:  # pragma: no cover
            raise ImportError(
                "ultralytics is not installed. Run: pip install ultralytics"
            ) from ex

        self.model = YOLO(model_path)
        self.confidence = confidence
        self.iou = iou
        self.classes = classes if classes is not None else [2, 3, 5, 7]
        # Set from the lanes with detection_areas(); without them the whole frame is searched.
        self.main_area: Area | None = None
        self.far_areas: list[Area] = []

    def _boxes_in(self, frame: np.ndarray, area: Area | None, image_size: int) -> list[tuple[Area, float]]:
        """Vehicle boxes (in frame coordinates) found in one area of the frame."""
        x0, y0, x1, y1 = area or (0, 0, frame.shape[1], frame.shape[0])
        # The area is cut out but nothing in it is blacked out: the lanes only filter the results, because most
        # of a vehicle stands above the road it drives on.
        # agnostic_nms: one vehicle detected as both e.g. "car" and "truck" must yield a single box,
        # otherwise it becomes two tracks and gets measured twice.
        results = self.model.predict(
            source=frame[y0:y1, x0:x1],
            conf=self.confidence,
            iou=self.iou,
            classes=self.classes,
            agnostic_nms=True,
            verbose=False,
            imgsz=image_size,
        )
        if not results or results[0].boxes is None:
            return []
        boxes: list[tuple[Area, float]] = []
        for box in results[0].boxes:
            bx1, by1, bx2, by2 = [int(v) for v in box.xyxy[0].tolist()]
            score = float(box.conf[0].item()) if box.conf is not None else 1.0
            boxes.append(((bx1 + x0, by1 + y0, bx2 + x0, by2 + y0), score))
        return boxes

    def detect(self, frame: np.ndarray, roi_mask: np.ndarray | None = None) -> List[Detection]:
        boxes = self._boxes_in(frame, self.main_area, MAIN_IMAGE_SIZE)
        for area in self.far_areas:
            longest = max(area[2] - area[0], area[3] - area[1])
            size = min(FAR_IMAGE_SIZE, -(-longest // 32) * 32)  # full resolution, in the detector's steps of 32
            for box, score in self._boxes_in(frame, area, size):
                # A vehicle leaving the area is cut off at its edge, which would put its road point on the edge
                # instead of on the road; the first look sees that vehicle whole.
                cut_off = (
                    (box[0] <= area[0] + 2 and area[0] > 0)
                    or (box[2] >= area[2] - 2 and area[2] < frame.shape[1])
                    or (box[3] >= area[3] - 2 and area[3] < frame.shape[0])
                )
                if cut_off:
                    continue
                # The closer look places the box more exactly, so it replaces the first look's box of that vehicle.
                boxes = [(b, s) for b, s in boxes if _overlap_of_smaller(b, box) < 0.5]
                boxes.append((box, score))

        detections: list[Detection] = []
        for bbox, score in boxes:
            if roi_mask is not None:
                x, y = road_point(bbox)
                if roi_mask[min(y, roi_mask.shape[0] - 1), min(x, roi_mask.shape[1] - 1)] == 0:
                    continue
            detections.append(Detection(bbox=bbox, score=score, class_name="vehicle"))
        return detections
