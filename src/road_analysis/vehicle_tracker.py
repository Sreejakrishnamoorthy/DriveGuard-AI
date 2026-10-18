"""
Vehicle Tracker for Road Analysis.

Assigns persistent tracking IDs to detected objects across video frames
using ByteTrack, a real-time multi-object tracking algorithm.

Implementation note:
    ByteTrack is provided by the `supervision` library (Roboflow), which
    ships a maintained implementation of the ByteTrack algorithm
    (Zhang et al., 2022). We do not reimplement ByteTrack from scratch;
    we wrap the library's tracker behind a clean, modular interface so it
    can be swapped out later without touching the rest of the codebase.
"""

import warnings
from dataclasses import dataclass
from typing import List, Tuple
import cv2
import numpy as np

from src.road_analysis.vehicle_detector import Detection
from src.utils.logger import logger

# supervision emits a FutureWarning because ByteTrack access moved to a
# submodule in recent versions; the class remains fully functional.
warnings.filterwarnings("ignore", category=FutureWarning, message=".*ByteTrack.*")

try:
    import supervision as sv

    SUPERVISION_AVAILABLE = True
except ImportError:
    SUPERVISION_AVAILABLE = False
    logger.warning(
        "supervision package not installed. Vehicle tracking will be unavailable. "
        "Install with: pip install supervision"
    )


@dataclass
class TrackedObject:
    """Data class for a tracked object with a persistent ID."""

    tracking_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    center_x: float
    center_y: float

    def width(self) -> float:
        """Get bounding box width."""
        x1, _, x2, _ = self.bbox
        return x2 - x1

    def height(self) -> float:
        """Get bounding box height."""
        _, y1, _, y2 = self.bbox
        return y2 - y1


class VehicleTracker:
    """
    Tracks detected vehicles/pedestrians across video frames using ByteTrack.

    ByteTrack associates detections between frames using a two-stage matching
    strategy (high-confidence detections first, then low-confidence ones to
    recover partially occluded objects), combined with Kalman-filter motion
    prediction. This gives each object a stable ID across frames instead of
    the ID changing every time raw detection is re-run.

    KNOWN LIMITATION (verified during development): the underlying
    `supervision.ByteTrack` implementation returns zero tracked objects on
    the frame immediately following any frame with zero detections — even
    if that next frame has real, valid detections. Tracking self-corrects
    on the following frame. In practice this is a single self-correcting
    frame gap (~33ms at 30fps) any time the scene briefly goes from "no
    vehicles visible" to "a vehicle appears," which is common in real
    driving footage (e.g. a car briefly out of frame). It was reproduced
    consistently against the raw library directly, independent of this
    wrapper's code, so it is not something a change here can fix; noting
    it here so it isn't mistaken for a bug in this project's own logic if
    a single frame's vehicle count looks unexpectedly low.
    """

    def __init__(
        self,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
        frame_rate: int = 30,
    ) -> None:
        """
        Initialize the vehicle tracker.

        Args:
            track_activation_threshold: Min confidence to start a new track
            lost_track_buffer: Frames to keep a lost track alive before dropping it
            minimum_matching_threshold: IoU threshold for matching detections to tracks
            frame_rate: Expected video frame rate (affects track buffer timing)
        """
        self.track_activation_threshold = track_activation_threshold
        self.lost_track_buffer = lost_track_buffer
        self.minimum_matching_threshold = minimum_matching_threshold
        self.frame_rate = frame_rate

        self.tracker = None
        self._class_name_by_id: dict = {}

        if not SUPERVISION_AVAILABLE:
            logger.error("Cannot initialize VehicleTracker: supervision package missing.")
            return

        self._initialize_tracker()

    def _initialize_tracker(self) -> None:
        """Create the underlying ByteTrack tracker instance."""
        try:
            with warnings.catch_warnings():
                # Scoped suppression (module-level filterwarnings can be
                # reset by test runners like pytest, so suppress right at
                # the call site to reliably silence this known, harmless
                # deprecation notice regardless of the calling context).
                warnings.filterwarnings("ignore", category=FutureWarning, message=".*ByteTrack.*")
                self.tracker = sv.ByteTrack(
                    track_activation_threshold=self.track_activation_threshold,
                    lost_track_buffer=self.lost_track_buffer,
                    minimum_matching_threshold=self.minimum_matching_threshold,
                    frame_rate=self.frame_rate,
                )
            logger.info("VehicleTracker (ByteTrack) initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize ByteTrack: {e}")
            self.tracker = None

    def _detections_to_sv_format(self, detections: List[Detection]) -> "sv.Detections":
        """
        Convert internal Detection objects to supervision's Detections format.

        Args:
            detections: List of Detection objects from VehicleDetector

        Returns:
            sv.Detections object
        """
        if not detections:
            return sv.Detections.empty()

        xyxy = np.array([d.bbox for d in detections], dtype=np.float32)
        confidence = np.array([d.confidence for d in detections], dtype=np.float32)
        class_id = np.array([d.class_id for d in detections], dtype=int)

        # Remember class names by class_id so we can restore them after tracking
        for d in detections:
            self._class_name_by_id[d.class_id] = d.class_name

        return sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)

    def update(self, detections: List[Detection]) -> List[TrackedObject]:
        """
        Update tracker with new detections and return tracked objects.

        Args:
            detections: List of Detection objects for the current frame

        Returns:
            List of TrackedObject with persistent tracking IDs.
            Returns an empty list if the tracker is unavailable.
        """
        if self.tracker is None:
            logger.debug("Tracker unavailable; returning empty tracking result.")
            return []

        try:
            sv_detections = self._detections_to_sv_format(detections)
            tracked_detections = self.tracker.update_with_detections(sv_detections)

            tracked_objects = []

            if tracked_detections.tracker_id is None:
                return []

            for i in range(len(tracked_detections)):
                x1, y1, x2, y2 = tracked_detections.xyxy[i]
                tracking_id = int(tracked_detections.tracker_id[i])
                class_id = int(tracked_detections.class_id[i])
                confidence = float(tracked_detections.confidence[i])

                class_name = self._class_name_by_id.get(class_id, "unknown")
                center_x = float((x1 + x2) / 2)
                center_y = float((y1 + y2) / 2)

                tracked_objects.append(
                    TrackedObject(
                        tracking_id=tracking_id,
                        class_name=class_name,
                        confidence=confidence,
                        bbox=(float(x1), float(y1), float(x2), float(y2)),
                        center_x=center_x,
                        center_y=center_y,
                    )
                )

            return tracked_objects

        except Exception as e:
            logger.error(f"Error during tracking update: {e}", exc_info=True)
            return []

    def draw_tracks(
        self,
        frame: np.ndarray,
        tracked_objects: List[TrackedObject],
        color: Tuple[int, int, int] = (0, 200, 255),
        thickness: int = 2,
    ) -> np.ndarray:
        """
        Draw tracked objects with their tracking IDs on the frame.

        Args:
            frame: Input frame
            tracked_objects: List of tracked objects
            color: Box color (BGR)
            thickness: Box line thickness

        Returns:
            Annotated frame
        """
        annotated = frame.copy()

        for obj in tracked_objects:
            x1, y1, x2, y2 = [int(v) for v in obj.bbox]

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            label = f"ID:{obj.tracking_id} {obj.class_name}"
            label_size, baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            label_y = max(y1, label_size[1] + 5)
            cv2.rectangle(
                annotated,
                (x1, label_y - label_size[1] - 5),
                (x1 + label_size[0], label_y + baseline - 5),
                color,
                -1,
            )
            cv2.putText(
                annotated,
                label,
                (x1, label_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
            )

            # Draw center point (useful for following-distance / lane logic)
            cv2.circle(annotated, (int(obj.center_x), int(obj.center_y)), 3, color, -1)

        return annotated

    def reset(self) -> None:
        """Reset tracker state (e.g., when starting a new video)."""
        if self.tracker is not None:
            self.tracker.reset()
            logger.info("VehicleTracker state reset.")

    def is_available(self) -> bool:
        """Check if the tracker is functional."""
        return self.tracker is not None
