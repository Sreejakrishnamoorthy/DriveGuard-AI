"""
Road Analyzer.

Integrates VehicleDetector, VehicleTracker, LaneDetector, and HazardDetector
into a single per-frame road-intelligence pipeline, and produces a combined
annotated frame plus structured output for the risk engine.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import time
import cv2
import numpy as np

from src.road_analysis.vehicle_detector import VehicleDetector
from src.road_analysis.vehicle_tracker import VehicleTracker, TrackedObject
from src.road_analysis.lane_detector import LaneDetector, LaneResult
from src.road_analysis.hazard_detector import HazardDetector, HazardResult
from src.road_analysis.near_miss_detector import NearMissDetector, NearMissResult, NEAR_MISS_NONE
from src.utils.logger import logger

# Object classes counted as "vehicles ahead" for following-distance purposes
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle"}

FOLLOWING_DISTANCE_SAFE = "SAFE"
FOLLOWING_DISTANCE_MODERATE = "MODERATE"
FOLLOWING_DISTANCE_UNSAFE = "UNSAFE"


@dataclass
class RoadAnalysisResult:
    """Combined result of one frame's road analysis."""

    vehicles: List[TrackedObject] = field(default_factory=list)
    lane: Optional[LaneResult] = None
    hazards: Optional[HazardResult] = None
    following_distance: str = FOLLOWING_DISTANCE_SAFE
    vehicle_summary: dict = field(default_factory=dict)
    near_miss: Optional[NearMissResult] = None

    def to_dict(self) -> dict:
        """Convert to the plain dict shape described in the project spec."""
        return {
            "vehicles": [
                {
                    "tracking_id": v.tracking_id,
                    "class_name": v.class_name,
                    "confidence": round(v.confidence, 3),
                    "bbox": [round(x, 1) for x in v.bbox],
                    "center_x": round(v.center_x, 1),
                    "center_y": round(v.center_y, 1),
                }
                for v in self.vehicles
            ],
            "lane": self.lane.to_dict() if self.lane else {
                "lane_detected": False,
                "lane_status": "UNKNOWN",
                "lane_departure": False,
            },
            "hazards": self.hazards.to_dict()["hazards"] if self.hazards else [],
            "hazard_detected": self.hazards.hazard_detected if self.hazards else False,
            "following_distance": self.following_distance,
            "vehicle_summary": self.vehicle_summary,
            "near_miss": self.near_miss.to_dict() if self.near_miss else {
                "near_miss_level": NEAR_MISS_NONE,
                "time_to_collision_seconds": None,
            },
        }


class RoadAnalyzer:
    """
    Combines vehicle detection, tracking, lane detection, and hazard detection
    into a single road-analysis pipeline.

    One instance should persist across a video's frames (not be recreated
    per frame), since VehicleTracker and LaneDetector maintain temporal
    state (track IDs, offset smoothing) between calls.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.40,
        hazard_model_path: Optional[str] = None,
        following_distance_ego_lane_ratio: float = 0.5,
        following_distance_unsafe_ratio: float = 0.45,
        following_distance_moderate_ratio: float = 0.22,
        roi_top_ratio: float = 0.6,
        roi_bottom_ratio: float = 1.0,
    ) -> None:
        """
        Initialize the road analyzer and its sub-components.

        Args:
            confidence_threshold: Min confidence for vehicle detections
            hazard_model_path: Optional path to a pretrained hazard model
            following_distance_ego_lane_ratio: Fraction of frame width (centered)
                considered "directly ahead" when estimating following distance
            following_distance_unsafe_ratio: Bbox-height/frame-height ratio above
                which the vehicle ahead is considered UNSAFE-close
            following_distance_moderate_ratio: Ratio above which it's MODERATE
            roi_top_ratio: Fraction of frame height above which lane/hazard
                detection ignores (sky/horizon). Shared by both sub-detectors
                so their road regions stay consistent.
            roi_bottom_ratio: Fraction of frame height BELOW which lane/hazard
                detection ignores. Default 1.0 assumes the road is visible to
                the bottom edge of the frame. IMPORTANT: if your camera is
                mounted low enough that the hood/dashboard is visible in the
                lower part of the frame, LOWER this value (e.g. 0.55-0.6) so
                the detectors stop before the hood — otherwise they will
                analyze hood reflections instead of the actual road surface.
                Check a sample frame from your footage first to find where
                the hood begins, as a fraction of frame height.
        """
        logger.info("Initializing RoadAnalyzer...")

        self.vehicle_detector = VehicleDetector(confidence_threshold=confidence_threshold)
        self.vehicle_tracker = VehicleTracker()
        self.lane_detector = LaneDetector(
            roi_top_ratio=roi_top_ratio, roi_bottom_ratio=roi_bottom_ratio
        )
        self.hazard_detector = HazardDetector(
            model_path=hazard_model_path,
            roi_top_ratio=roi_top_ratio,
            roi_bottom_ratio=roi_bottom_ratio,
        )
        self.near_miss_detector = NearMissDetector()

        self.ego_lane_ratio = following_distance_ego_lane_ratio
        self.unsafe_ratio = following_distance_unsafe_ratio
        self.moderate_ratio = following_distance_moderate_ratio

        logger.info("RoadAnalyzer initialized successfully.")
        logger.info(f"Hazard detector mode: {self.hazard_detector.mode}")

    def _estimate_following_distance(
        self, tracked_objects: List[TrackedObject], frame_shape: Tuple[int, int, int]
    ) -> str:
        """
        Estimate relative following distance to the vehicle directly ahead.

        IMPORTANT — monocular distance limitation:
            This is a RELATIVE, monocular-camera heuristic, not a calibrated
            real-world distance measurement. Without stereo cameras, LiDAR,
            or a calibrated camera model (known focal length + real vehicle
            dimensions), absolute distance in meters cannot be recovered
            from a single 2D image. This method instead uses a well-known
            perspective proxy: as a vehicle ahead gets closer, its bounding
            box occupies a larger fraction of the frame's height. That ratio
            is bucketed into SAFE / MODERATE / UNSAFE. Treat these labels as
            *relative* risk indicators, not measured distances.

        Args:
            tracked_objects: Currently tracked objects
            frame_shape: (height, width, channels) of the frame

        Returns:
            One of "SAFE", "MODERATE", "UNSAFE"
        """
        height, width = frame_shape[:2]

        lane_left = width * (0.5 - self.ego_lane_ratio / 2)
        lane_right = width * (0.5 + self.ego_lane_ratio / 2)

        candidates = [
            obj
            for obj in tracked_objects
            if obj.class_name in VEHICLE_CLASSES and lane_left <= obj.center_x <= lane_right
        ]

        if not candidates:
            # Nothing ahead in the ego lane -> nothing to be unsafe about
            return FOLLOWING_DISTANCE_SAFE

        # The vehicle occupying the largest vertical fraction of the frame
        # is treated as the closest one directly ahead.
        closest = max(candidates, key=lambda o: o.height())
        height_ratio = closest.height() / height

        if height_ratio >= self.unsafe_ratio:
            return FOLLOWING_DISTANCE_UNSAFE
        elif height_ratio >= self.moderate_ratio:
            return FOLLOWING_DISTANCE_MODERATE
        else:
            return FOLLOWING_DISTANCE_SAFE

    @staticmethod
    def _build_vehicle_summary(tracked_objects: List[TrackedObject]) -> dict:
        """Summarize tracked objects by category for dashboard display."""
        summary = {
            "total": len(tracked_objects),
            "cars": 0,
            "trucks": 0,
            "buses": 0,
            "motorcycles": 0,
            "bicycles": 0,
            "pedestrians": 0,
        }

        class_map = {
            "car": "cars",
            "truck": "trucks",
            "bus": "buses",
            "motorcycle": "motorcycles",
            "bicycle": "bicycles",
            "person": "pedestrians",
        }

        for obj in tracked_objects:
            key = class_map.get(obj.class_name)
            if key:
                summary[key] += 1

        return summary

    def analyze(self, frame: np.ndarray, timestamp: Optional[float] = None) -> RoadAnalysisResult:
        """
        Run the full road-analysis pipeline on a single frame.

        Args:
            frame: Input frame (BGR)
            timestamp: Optional explicit timestamp (seconds). Pass this
                when processing a video file so near-miss/TTC timing
                reflects the VIDEO's timeline rather than wall-clock
                processing speed. Defaults to wall-clock time if omitted.

        Returns:
            RoadAnalysisResult with vehicles, lane, hazards, following
            distance, and near-miss/TTC assessment
        """
        if timestamp is None:
            timestamp = time.time()

        try:
            detections = self.vehicle_detector.detect(frame)
            tracked_objects = self.vehicle_tracker.update(detections)
            lane_result = self.lane_detector.detect(frame)
            hazard_result = self.hazard_detector.detect(frame)
            following_distance = self._estimate_following_distance(
                tracked_objects, frame.shape
            )
            vehicle_summary = self._build_vehicle_summary(tracked_objects)
            near_miss_result = self.near_miss_detector.update(
                tracked_objects, frame.shape[1], timestamp
            )

            return RoadAnalysisResult(
                vehicles=tracked_objects,
                lane=lane_result,
                hazards=hazard_result,
                following_distance=following_distance,
                vehicle_summary=vehicle_summary,
                near_miss=near_miss_result,
            )

        except Exception as e:
            logger.error(f"RoadAnalyzer: error during analysis: {e}", exc_info=True)
            return RoadAnalysisResult()

    def draw_annotations(self, frame: np.ndarray, result: RoadAnalysisResult) -> np.ndarray:
        """
        Draw all road-analysis layers (lane, hazards, tracked vehicles) onto one frame.

        Args:
            frame: Original frame
            result: RoadAnalysisResult from analyze()

        Returns:
            Fully annotated frame
        """
        annotated = frame

        if result.lane is not None:
            annotated = self.lane_detector.draw_lanes(annotated, result.lane)

        if result.hazards is not None:
            annotated = self.hazard_detector.draw_hazards(annotated, result.hazards)

        annotated = self.vehicle_tracker.draw_tracks(annotated, result.vehicles)

        # Following-distance status banner
        fd_color = {
            FOLLOWING_DISTANCE_SAFE: (0, 200, 0),
            FOLLOWING_DISTANCE_MODERATE: (0, 165, 255),
            FOLLOWING_DISTANCE_UNSAFE: (0, 0, 255),
        }.get(result.following_distance, (200, 200, 200))

        cv2.putText(
            annotated,
            f"Following Distance: {result.following_distance}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            fd_color,
            2,
        )

        if result.near_miss is not None and result.near_miss.near_miss_level != NEAR_MISS_NONE:
            nm_color = (0, 0, 255) if result.near_miss.near_miss_level == "CRITICAL" else (0, 140, 255)
            ttc = result.near_miss.time_to_collision_seconds
            ttc_text = f"{ttc:.1f}s" if ttc is not None else "?"
            cv2.putText(
                annotated,
                f"NEAR MISS ({result.near_miss.near_miss_level}) - est. TTC {ttc_text}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                nm_color,
                2,
            )

        return annotated

    def reset(self) -> None:
        """Reset temporal state in sub-components (e.g., for a new video)."""
        self.vehicle_tracker.reset()
        self.lane_detector.reset()
        self.near_miss_detector.reset()
        logger.info("RoadAnalyzer state reset.")
