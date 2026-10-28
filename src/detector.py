from __future__ import annotations

from typing import List

import cv2
import numpy as np

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

    def detect(self, frame: np.ndarray, roi_mask: np.ndarray | None = None) -> List[Detection]:
        if roi_mask is not None:
            masked = cv2.bitwise_and(frame, frame, mask=roi_mask)
        else:
            masked = frame

        # agnostic_nms: one vehicle detected as both e.g. "car" and "truck" must yield a single box,
        # otherwise it becomes two tracks and gets measured twice.
        results = self.model.predict(
            source=masked,
            conf=self.confidence,
            iou=self.iou,
            classes=self.classes,
            agnostic_nms=True,
            verbose=False,
        )

        detections: list[Detection] = []
        if not results:
            return detections

        result = results[0]
        if result.boxes is None:
            return detections

        for box in result.boxes:
            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            score = float(box.conf[0].item()) if box.conf is not None else 1.0
            detections.append(Detection(bbox=(x1, y1, x2, y2), score=score, class_name="vehicle"))

        return detections
