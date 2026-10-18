"""
Eye Detector for Driver Monitoring.

Estimates per-frame eye state (OPEN/CLOSED) using the Eye Aspect Ratio
(EAR) technique (Soukupova & Cech, 2016), computed from MediaPipe Face
Mesh landmarks. This module only looks at a SINGLE frame — temporal
reasoning (how long the eyes have been closed) belongs in
DrowsinessDetector, which consumes this module's output over time.
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
    logger.warning("mediapipe package not installed. Eye detection will be unavailable.")

# Standard MediaPipe Face Mesh landmark indices for Eye Aspect Ratio.
# Each list is ordered [left_corner, upper_1, upper_2, right_corner, lower_2, lower_1]
# matching the classic 6-point EAR formulation.
RIGHT_EYE_EAR_IDX = [33, 160, 158, 133, 153, 144]
LEFT_EYE_EAR_IDX = [362, 385, 387, 263, 373, 380]

EYE_STATE_OPEN = "OPEN"
EYE_STATE_CLOSED = "CLOSED"
EYE_STATE_UNKNOWN = "UNKNOWN"


@dataclass
class EyeState:
    """Result of eye-state estimation for a single frame."""

    eye_state: str  # OPEN, CLOSED, UNKNOWN
    ear_value: Optional[float] = None
    left_ear: Optional[float] = None
    right_ear: Optional[float] = None
    landmarks: Optional[object] = None  # raw mediapipe landmarks, for reuse by AttentionDetector

    def to_dict(self) -> dict:
        return {
            "eye_state": self.eye_state,
            "ear_value": round(self.ear_value, 4) if self.ear_value is not None else None,
        }


def _euclidean(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """2D Euclidean distance between two points."""
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def compute_ear(eye_points: List[Tuple[float, float]]) -> float:
    """
    Compute Eye Aspect Ratio from 6 eye landmark points.

    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    Higher EAR (~0.25-0.35 typical) indicates an open eye; EAR dropping
    sharply toward ~0.1-0.15 indicates a closed eye. Exact values vary by
    face geometry and camera angle, which is why classification uses a
    configurable threshold rather than a fixed universal cutoff.

    Args:
        eye_points: 6 (x, y) points in order [corner, up1, up2, corner, low2, low1]

    Returns:
        EAR value (float)
    """
    p1, p2, p3, p4, p5, p6 = eye_points
    vertical_1 = _euclidean(p2, p6)
    vertical_2 = _euclidean(p3, p5)
    horizontal = _euclidean(p1, p4)

    if horizontal == 0:
        return 0.0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


class EyeDetector:
    """
    Estimates eye-open/closed state for a single frame using MediaPipe
    Face Mesh landmarks and the Eye Aspect Ratio (EAR) technique.
    """

    def __init__(
        self,
        ear_threshold: float = 0.21,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        face_mesh: Optional[object] = None,
    ) -> None:
        """
        Initialize eye detector.

        Args:
            ear_threshold: EAR below this value is classified as CLOSED.
                0.21 is a commonly-used starting point in EAR literature;
                tune per-camera/lighting if you observe misclassification.
            min_detection_confidence: MediaPipe face detection confidence
            min_tracking_confidence: MediaPipe landmark tracking confidence
            face_mesh: Optionally inject an existing MediaPipe FaceMesh
                instance (e.g. from ModelManager.get_face_mesh()) so this
                detector reuses it instead of creating its own — avoids
                running the face-mesh model twice per frame when both
                EyeDetector and AttentionDetector are used together (see
                DriverAnalyzer). If None, creates and owns its own instance
                for standalone use.
        """
        self.ear_threshold = ear_threshold
        self._owns_face_mesh = face_mesh is None
        self._face_mesh = face_mesh

        if self._face_mesh is None and MEDIAPIPE_AVAILABLE:
            try:
                self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence,
                )
                logger.info("EyeDetector: MediaPipe FaceMesh model loaded.")
            except Exception as e:
                logger.error(f"EyeDetector: failed to load FaceMesh model: {e}")
                self._face_mesh = None

    def _extract_eye_points(
        self, landmarks, indices: List[int], width: int, height: int
    ) -> List[Tuple[float, float]]:
        """Convert normalized MediaPipe landmarks to pixel coordinates for given indices."""
        return [
            (landmarks[i].x * width, landmarks[i].y * height) for i in indices
        ]

    def analyze_landmarks(self, landmarks, width: int, height: int) -> EyeState:
        """
        Compute eye state from already-extracted MediaPipe face landmarks.

        Use this directly (instead of detect()) when landmarks were already
        computed elsewhere in the same frame (e.g. by DriverAnalyzer sharing
        one FaceMesh inference across EyeDetector and AttentionDetector).

        Args:
            landmarks: MediaPipe NormalizedLandmarkList.landmark (indexable)
            width: Frame width in pixels
            height: Frame height in pixels

        Returns:
            EyeState with eye_state classification and EAR values
        """
        try:
            right_points = self._extract_eye_points(landmarks, RIGHT_EYE_EAR_IDX, width, height)
            left_points = self._extract_eye_points(landmarks, LEFT_EYE_EAR_IDX, width, height)

            right_ear = compute_ear(right_points)
            left_ear = compute_ear(left_points)
            avg_ear = (right_ear + left_ear) / 2.0

            state = EYE_STATE_CLOSED if avg_ear < self.ear_threshold else EYE_STATE_OPEN

            return EyeState(
                eye_state=state,
                ear_value=avg_ear,
                left_ear=left_ear,
                right_ear=right_ear,
                landmarks=landmarks,
            )
        except Exception as e:
            logger.error(f"EyeDetector: error analyzing landmarks: {e}", exc_info=True)
            return EyeState(eye_state=EYE_STATE_UNKNOWN)

    def detect(self, frame: np.ndarray) -> EyeState:
        """
        Estimate eye state for the current frame (standalone use — runs its
        own FaceMesh inference). For shared-inference use, see analyze_landmarks().

        Args:
            frame: Input frame (BGR)

        Returns:
            EyeState with eye_state classification and EAR values.
            Returns UNKNOWN if no face/landmarks are found or MediaPipe
            is unavailable — callers should treat UNKNOWN as "no signal",
            not as "eyes open".
        """
        if self._face_mesh is None:
            return EyeState(eye_state=EYE_STATE_UNKNOWN)

        try:
            height, width = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self._face_mesh.process(rgb)

            if not results.multi_face_landmarks:
                return EyeState(eye_state=EYE_STATE_UNKNOWN)

            landmarks = results.multi_face_landmarks[0].landmark
            return self.analyze_landmarks(landmarks, width, height)

        except Exception as e:
            logger.error(f"EyeDetector: error during detection: {e}", exc_info=True)
            return EyeState(eye_state=EYE_STATE_UNKNOWN)

    def draw_eyes(self, frame: np.ndarray, state: EyeState) -> np.ndarray:
        """Draw EAR value and eye-state label on frame."""
        annotated = frame.copy()
        if state.eye_state != EYE_STATE_UNKNOWN:
            color = (0, 200, 0) if state.eye_state == EYE_STATE_OPEN else (0, 0, 255)
            label = f"Eyes: {state.eye_state} (EAR {state.ear_value:.2f})"
            cv2.putText(
                annotated, label, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )
        return annotated

    def is_available(self) -> bool:
        """Whether the underlying model loaded successfully."""
        return self._face_mesh is not None
