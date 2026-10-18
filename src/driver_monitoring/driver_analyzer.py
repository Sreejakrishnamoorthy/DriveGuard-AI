"""
Driver Analyzer.

Integrates FaceDetector, EyeDetector, DrowsinessDetector, and
AttentionDetector into a single per-frame driver-monitoring pipeline.

Performance note: EyeDetector and AttentionDetector both need MediaPipe
Face Mesh landmarks. Rather than each running its own FaceMesh inference
(two model calls per frame), this analyzer runs FaceMesh ONCE via the
shared ModelManager instance and passes the same landmarks to both.
"""

from dataclasses import dataclass
from typing import Optional
import cv2
import numpy as np

from src.driver_monitoring.face_detector import FaceDetector, FaceDetectionResult
from src.driver_monitoring.eye_detector import EyeDetector, EyeState, EYE_STATE_UNKNOWN
from src.driver_monitoring.drowsiness_detector import DrowsinessDetector, DrowsinessResult
from src.driver_monitoring.attention_detector import (
    AttentionDetector,
    AttentionResult,
    ATTENTION_UNKNOWN,
    ATTENTION_DISTRACTED,
)
from src.models.model_manager import model_manager
from src.utils.logger import logger

DRIVER_STATUS_NORMAL = "NORMAL"
DRIVER_STATUS_DROWSY = "DROWSY"
DRIVER_STATUS_DISTRACTED = "DISTRACTED"
DRIVER_STATUS_DROWSY_AND_DISTRACTED = "DROWSY_AND_DISTRACTED"
DRIVER_STATUS_NO_FACE_DETECTED = "NO_FACE_DETECTED"


@dataclass
class DriverAnalysisResult:
    """Combined result of one frame's driver analysis."""

    face_detected: bool
    face_bbox: Optional[tuple] = None
    face_confidence: float = 0.0
    eye_state: str = EYE_STATE_UNKNOWN
    ear_value: Optional[float] = None
    drowsiness: bool = False
    drowsiness_probability: float = 0.0
    attention: str = ATTENTION_UNKNOWN
    pose_yaw: Optional[float] = None
    pose_pitch: Optional[float] = None
    driver_status: str = DRIVER_STATUS_NO_FACE_DETECTED

    def to_dict(self) -> dict:
        """Convert to the plain dict shape described in the project spec."""
        return {
            "face_detected": self.face_detected,
            "eye_state": self.eye_state,
            "drowsiness": self.drowsiness,
            "drowsiness_probability": round(self.drowsiness_probability, 3),
            "attention": self.attention,
            "driver_status": self.driver_status,
        }


class DriverAnalyzer:
    """
    Combines face detection, eye state, drowsiness, and attention into a
    single driver-monitoring pipeline.

    One instance should persist across a video's frames (not be recreated
    per frame), since DrowsinessDetector maintains temporal state between
    calls.
    """

    def __init__(
        self,
        ear_threshold: float = 0.21,
        closed_duration_threshold_seconds: float = 1.5,
        perclos_window_seconds: float = 10.0,
        perclos_threshold: float = 0.4,
        yaw_threshold_degrees: float = 30.0,
        pitch_threshold_degrees: float = 25.0,
    ) -> None:
        """Initialize the driver analyzer and its sub-components."""
        logger.info("Initializing DriverAnalyzer...")

        self.face_detector = FaceDetector()

        # Share one FaceMesh inference across eye + attention detection
        shared_face_mesh = model_manager.get_face_mesh(refine_landmarks=True)

        self.eye_detector = EyeDetector(
            ear_threshold=ear_threshold, face_mesh=shared_face_mesh
        )
        self.attention_detector = AttentionDetector(
            yaw_threshold_degrees=yaw_threshold_degrees,
            pitch_threshold_degrees=pitch_threshold_degrees,
            face_mesh=shared_face_mesh,
        )
        self.drowsiness_detector = DrowsinessDetector(
            closed_duration_threshold_seconds=closed_duration_threshold_seconds,
            perclos_window_seconds=perclos_window_seconds,
            perclos_threshold=perclos_threshold,
        )
        self._shared_face_mesh = shared_face_mesh

        logger.info("DriverAnalyzer initialized successfully.")

    @staticmethod
    def _determine_driver_status(drowsy: bool, attention: str) -> str:
        """Combine drowsiness and attention signals into one overall status."""
        distracted = attention == ATTENTION_DISTRACTED

        if drowsy and distracted:
            return DRIVER_STATUS_DROWSY_AND_DISTRACTED
        if drowsy:
            return DRIVER_STATUS_DROWSY
        if distracted:
            return DRIVER_STATUS_DISTRACTED
        return DRIVER_STATUS_NORMAL

    def analyze(self, frame: np.ndarray, timestamp: float = None) -> DriverAnalysisResult:
        """
        Run the full driver-monitoring pipeline on a single frame.

        Args:
            frame: Input frame (BGR), typically from the driver-facing camera
            timestamp: Optional explicit timestamp (seconds). Pass this when
                processing a video file so drowsiness timing reflects the
                VIDEO's timeline rather than wall-clock processing speed.

        Returns:
            DriverAnalysisResult with face, eye, drowsiness, and attention state
        """
        try:
            face_result = self.face_detector.detect(frame)

            if not face_result.face_detected:
                # Still update drowsiness temporal state with UNKNOWN so a
                # brief face-detection dropout doesn't corrupt consecutive-
                # closure tracking, but don't claim any eye/attention signal.
                drowsy_result = self.drowsiness_detector.update(EYE_STATE_UNKNOWN, timestamp)
                return DriverAnalysisResult(
                    face_detected=False,
                    driver_status=DRIVER_STATUS_NO_FACE_DETECTED,
                    drowsiness_probability=drowsy_result.drowsiness_probability,
                )

            height, width = frame.shape[:2]

            # Run FaceMesh ONCE and share landmarks between eye + attention
            # detection, instead of each running its own inference.
            eye_state = EyeState(eye_state=EYE_STATE_UNKNOWN)
            attention_result = AttentionResult(attention=ATTENTION_UNKNOWN)

            if self._shared_face_mesh is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mesh_results = self._shared_face_mesh.process(rgb)

                if mesh_results.multi_face_landmarks:
                    landmarks = mesh_results.multi_face_landmarks[0].landmark
                    eye_state = self.eye_detector.analyze_landmarks(landmarks, width, height)
                    attention_result = self.attention_detector.analyze_landmarks(
                        landmarks, width, height
                    )

            drowsy_result = self.drowsiness_detector.update(eye_state.eye_state, timestamp)

            driver_status = self._determine_driver_status(
                drowsy_result.drowsiness, attention_result.attention
            )

            return DriverAnalysisResult(
                face_detected=True,
                face_bbox=face_result.bbox,
                face_confidence=face_result.confidence,
                eye_state=eye_state.eye_state,
                ear_value=eye_state.ear_value,
                drowsiness=drowsy_result.drowsiness,
                drowsiness_probability=drowsy_result.drowsiness_probability,
                attention=attention_result.attention,
                pose_yaw=attention_result.yaw,
                pose_pitch=attention_result.pitch,
                driver_status=driver_status,
            )

        except Exception as e:
            logger.error(f"DriverAnalyzer: error during analysis: {e}", exc_info=True)
            return DriverAnalysisResult(face_detected=False)

    def draw_annotations(self, frame: np.ndarray, result: DriverAnalysisResult) -> np.ndarray:
        """Draw face box, driver status, and metrics onto the frame."""
        annotated = frame.copy()

        if result.face_detected and result.face_bbox:
            x1, y1, x2, y2 = result.face_bbox
            color = {
                DRIVER_STATUS_NORMAL: (0, 200, 0),
                DRIVER_STATUS_DROWSY: (0, 0, 255),
                DRIVER_STATUS_DISTRACTED: (0, 165, 255),
                DRIVER_STATUS_DROWSY_AND_DISTRACTED: (0, 0, 255),
            }.get(result.driver_status, (200, 200, 200))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        status_color = (0, 0, 255) if result.drowsiness else (0, 200, 0)
        cv2.putText(
            annotated,
            f"Driver: {result.driver_status}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            status_color,
            2,
        )
        cv2.putText(
            annotated,
            f"Eyes: {result.eye_state}  Attention: {result.attention}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        return annotated

    def reset(self) -> None:
        """Reset temporal state (e.g., when starting a new video/session)."""
        self.drowsiness_detector.reset()
        logger.info("DriverAnalyzer state reset.")
