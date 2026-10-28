from __future__ import annotations

from dataclasses import dataclass
from math import dist
from typing import Dict, List, Tuple

from .models import Detection, Track

Point = Tuple[int, int]


@dataclass
class TrackState:
    track: Track
    missing_frames: int = 0
    velocity: Tuple[float, float] = (0.0, 0.0)  # pixels per frame
    hits: int = 1


class CentroidTracker:
    def __init__(
        self,
        max_missing_frames: int = 20,
        max_match_distance_px: float = 75.0,
        velocity_smoothing: float = 0.5,
    ) -> None:
        self.max_missing_frames = max_missing_frames
        self.max_match_distance_px = max_match_distance_px
        self.velocity_smoothing = velocity_smoothing
        self.next_track_id = 1
        self.tracks: Dict[int, TrackState] = {}

    @staticmethod
    def _centroid_from_bbox(bbox: tuple[int, int, int, int]) -> Point:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @staticmethod
    def _predicted_centroid(state: TrackState) -> tuple[float, float]:
        # Constant-velocity guess of where the vehicle is now, so a track that missed a few
        # detections can still be matched instead of being replaced by a new track ID.
        cx, cy = state.track.centroid
        vx, vy = state.velocity
        return (cx + vx * state.missing_frames, cy + vy * state.missing_frames)

    def _match(self, detections: list[Detection]) -> list[tuple[int, int]]:
        matches: list[tuple[int, int]] = []
        used_tracks: set[int] = set()
        used_detections: set[int] = set()

        track_items = list(self.tracks.items())
        candidates: list[tuple[float, int, int]] = []

        for ti, (_, state) in enumerate(track_items):
            track_center = self._predicted_centroid(state)
            for di, detection in enumerate(detections):
                det_center = self._centroid_from_bbox(detection.bbox)
                d = dist(track_center, det_center)
                if d <= self.max_match_distance_px:
                    candidates.append((d, ti, di))

        candidates.sort(key=lambda x: x[0])

        for _, ti, di in candidates:
            track_id = track_items[ti][0]
            if track_id in used_tracks or di in used_detections:
                continue
            used_tracks.add(track_id)
            used_detections.add(di)
            matches.append((track_id, di))

        return matches

    def update(self, detections: List[Detection], ts_s: float) -> List[Track]:
        for track_id in list(self.tracks.keys()):
            self.tracks[track_id].missing_frames += 1

        matches = self._match(detections)
        matched_detections = {di for _, di in matches}

        for track_id, di in matches:
            bbox = detections[di].bbox
            centroid = self._centroid_from_bbox(bbox)
            state = self.tracks[track_id]
            track = state.track

            frames_elapsed = state.missing_frames
            new_vx = (centroid[0] - track.centroid[0]) / frames_elapsed
            new_vy = (centroid[1] - track.centroid[1]) / frames_elapsed
            a = self.velocity_smoothing if state.hits > 1 else 1.0
            state.velocity = (a * new_vx + (1 - a) * state.velocity[0], a * new_vy + (1 - a) * state.velocity[1])
            state.hits += 1

            track.bbox = bbox
            track.centroid = centroid
            track.last_seen_ts = ts_s
            state.missing_frames = 0

        for di, detection in enumerate(detections):
            if di in matched_detections:
                continue
            bbox = detection.bbox
            centroid = self._centroid_from_bbox(bbox)
            track = Track(
                track_id=self.next_track_id,
                bbox=bbox,
                centroid=centroid,
                last_seen_ts=ts_s,
            )
            self.tracks[self.next_track_id] = TrackState(track=track)
            self.next_track_id += 1

        for track_id in list(self.tracks.keys()):
            if self.tracks[track_id].missing_frames > self.max_missing_frames:
                del self.tracks[track_id]

        return [state.track for state in self.tracks.values()]
