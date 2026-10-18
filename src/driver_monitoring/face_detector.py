"""
Face Detector for Driver Monitoring.

Uses MediaPipe's lightweight face-detection solution to locate the
driver's face in each frame. This is intentionally a separate, cheaper
model from the full face-mesh landmark model used by eye/attention
detection — most frames just need a fast "is a face present" check.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2
import numpy as np

from src.utils.logger import logger

try:
    import mediapipe as mp

    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logger.warning(
        "mediapipe package not installed. Face detection will be unavailable. "
        "Install with: pip install mediapipe"
    )


@dataclass
class FaceDetectionResult:
    """Result of face detection for a single frame."""

    face_detected: bool
    bbox: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2)
    confidence: float = 0.0

    def to_dict(self) -> dict:
        """Convert to the plain dict shape described in the project spec."""
        return {
            "face_detected": self.face_detected,
            "bbox": list(self.bbox) if self.bbox else None,
        }


class FaceDetector:
    """
    Detects the driver's face in a frame using MediaPipe Face Detection.

    This is a lightweight bounding-box detector (not the full landmark
    mesh). EyeDetector and AttentionDetector use a separate, richer
    face-mesh model since they need per-landmark geometry.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        model_selection: int = 0,
    ) -> None:
        """
        Initialize face detector.

        Args:
            min_detection_confidence: Minimum confidence to report a detection
            model_selection: 0 = short-range model (best for faces within ~2m,
                appropriate for an in-cabin driver camera), 1 = full-range model
        """
        self.min_detection_confidence = min_detection_confidence
        self.model_selection = model_selection
        self._detector = None

        if MEDIAPIPE_AVAILABLE:
            self._load_model()

    def _load_model(self) -> None:
        """Load the MediaPipe face detection model."""
        try:
            self._detector = mp.solutions.face_detection.FaceDetection(
                model_selection=self.model_selection,
                min_detection_confidence=self.min_detection_confidence,
            )
            logger.info("FaceDetector: MediaPipe face detection model loaded.")
        except Exception as e:
            logger.error(f"FaceDetector: failed to load model: {e}")
            self._detector = None

    def detect(self, frame: np.ndarray) -> FaceDetectionResult:
        """
        Detect a face in the frame.

        Args:
            frame: Input frame (BGR)

        Returns:
            FaceDetectionResult with face_detected flag and bbox if found.
            If MediaPipe is unavailable, always returns face_detected=False
            rather than raising, so the rest of the pipeline can continue.
        """
        if self._detector is None:
            return FaceDetectionResult(face_detected=False)

        try:
            height, width = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self._detector.process(rgb)

            if not results.detections:
                return FaceDetectionResult(face_detected=False)

            # Use the highest-confidence detection (driver monitoring
            # expects a single occupant in frame).
            best = max(results.detections, key=lambda d: d.score[0])
            rel_bbox = best.location_data.relative_bounding_box

            x1 = max(0, int(rel_bbox.xmin * width))
            y1 = max(0, int(rel_bbox.ymin * height))
            x2 = min(width, int((rel_bbox.xmin + rel_bbox.width) * width))
            y2 = min(height, int((rel_bbox.ymin + rel_bbox.height) * height))

            return FaceDetectionResult(
                face_detected=True,
                bbox=(x1, y1, x2, y2),
                confidence=float(best.score[0]),
            )

        except Exception as e:
            logger.error(f"FaceDetector: error during detection: {e}", exc_info=True)
            return FaceDetectionResult(face_detected=False)

    def draw_detection(self, frame: np.ndarray, result: FaceDetectionResult) -> np.ndarray:
        """Draw face bounding box on frame."""
        annotated = frame.copy()
        if result.face_detected and result.bbox:
            x1, y1, x2, y2 = result.bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 150), 2)
            cv2.putText(
                annotated,
                f"Face {result.confidence:.2f}",
                (x1, max(y1 - 8, 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 150),
                1,
            )
        return annotated

    def is_available(self) -> bool:
        """Whether the underlying model loaded successfully."""
        return self._detector is not None
