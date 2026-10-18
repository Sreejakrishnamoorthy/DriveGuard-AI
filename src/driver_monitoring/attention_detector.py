"""
Attention Detector for Driver Monitoring.

Estimates coarse head orientation (pitch/yaw/roll) from MediaPipe Face
Mesh landmarks using OpenCV's solvePnP against a generic anthropometric
3D face model, then classifies attention as ATTENTIVE / DISTRACTED /
UNKNOWN based on how far the head is turned from facing forward.

IMPORTANT — this estimates HEAD ORIENTATION, not eye gaze direction.
A driver can face forward while their eyes look elsewhere (or vice
versa); this module makes no claim about where the eyes are actually
looking, only where the head is pointed, which is a reasonable and
literature-supported proxy for attention but not a precise gaze tracker.

IMPORTANT — camera calibration:
    Without a calibrated camera (known focal length/lens distortion),
    the camera matrix used here is an approximation (focal length ~=
    frame width, principal point = frame center). This is standard
    practice for uncalibrated head-pose estimation and works well for
    coarse ATTENTIVE/DISTRACTED classification, but resulting angles
    should be treated as approximate, not precise.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import numpy as np

from src.utils.logger import logger

try:
    import mediapipe as mp

    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logger.warning("mediapipe package not installed. Attention detection will be unavailable.")

ATTENTION_ATTENTIVE = "ATTENTIVE"
ATTENTION_DISTRACTED = "DISTRACTED"
ATTENTION_UNKNOWN = "UNKNOWN"

# MediaPipe Face Mesh landmark indices for a standard 6-point head-pose
# model (nose tip, chin, eye corners, mouth corners) — a well-established
# subset used across OpenCV/MediaPipe head-pose-estimation implementations.
NOSE_TIP_IDX = 1
CHIN_IDX = 152
LEFT_EYE_CORNER_IDX = 33
RIGHT_EYE_CORNER_IDX = 263
LEFT_MOUTH_CORNER_IDX = 61
RIGHT_MOUTH_CORNER_IDX = 291

# Generic anthropometric 3D face model (approximate, arbitrary units),
# a widely-used approximation for solvePnP-based head pose estimation
# when no subject-specific 3D scan is available. Y-axis convention is
# chosen so that, under an identity rotation, the chin point projects
# BELOW the nose in image space (larger image-Y = further down the
# frame) to match standard image coordinates — verified against a
# controlled synthetic projection test.
_MODEL_POINTS_3D = np.array(
    [
        (0.0, 0.0, 0.0),  # Nose tip
        (0.0, 330.0, -65.0),  # Chin
        (-225.0, -170.0, -135.0),  # Left eye left corner
        (225.0, -170.0, -135.0),  # Right eye right corner
        (-150.0, 150.0, -125.0),  # Left mouth corner
        (150.0, 150.0, -125.0),  # Right mouth corner
    ],
    dtype=np.float64,
)


@dataclass
class AttentionResult:
    """Result of attention/head-pose estimation for a single frame."""

    attention: str  # ATTENTIVE, DISTRACTED, UNKNOWN
    yaw: Optional[float] = None
    pitch: Optional[float] = None
    roll: Optional[float] = None

    def to_dict(self) -> dict:
        return {"attention": self.attention}


def _rotation_vector_to_euler_angles(rotation_vector: np.ndarray) -> Tuple[float, float, float]:
    """Convert a solvePnP rotation vector to (pitch, yaw, roll) in degrees."""
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        pitch = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        yaw = np.arctan2(-rotation_matrix[2, 0], sy)
        roll = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        pitch = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        yaw = np.arctan2(-rotation_matrix[2, 0], sy)
        roll = 0.0

    return (np.degrees(pitch), np.degrees(yaw), np.degrees(roll))


class AttentionDetector:
    """
    Estimates driver attention (ATTENTIVE/DISTRACTED) from head pose,
    computed via solvePnP against MediaPipe Face Mesh landmarks.
    """

    def __init__(
        self,
        yaw_threshold_degrees: float = 30.0,
        pitch_threshold_degrees: float = 25.0,
        min_detection_confidence: float = 0.5,
        face_mesh: Optional[object] = None,
    ) -> None:
        """
        Initialize attention detector.

        Args:
            yaw_threshold_degrees: |yaw| beyond this is classified DISTRACTED
                (head turned left/right, e.g. looking at a phone/passenger)
            pitch_threshold_degrees: |pitch| beyond this is classified
                DISTRACTED (head tilted down/up, e.g. looking at a phone in lap)
            min_detection_confidence: MediaPipe face detection confidence
            face_mesh: Optionally inject an existing MediaPipe FaceMesh
                instance so this detector reuses it instead of creating its
                own — avoids a duplicate inference call per frame when used
                alongside EyeDetector (see DriverAnalyzer). If None, creates
                and owns its own instance for standalone use.
        """
        self.yaw_threshold = yaw_threshold_degrees
        self.pitch_threshold = pitch_threshold_degrees
        self._owns_face_mesh = face_mesh is None
        self._face_mesh = face_mesh

        if self._face_mesh is None and MEDIAPIPE_AVAILABLE:
            try:
                self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=1,
                    refine_landmarks=False,
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_detection_confidence,
                )
                logger.info("AttentionDetector: MediaPipe FaceMesh model loaded.")
            except Exception as e:
                logger.error(f"AttentionDetector: failed to load FaceMesh model: {e}")
                self._face_mesh = None

    def _estimate_pose_from_landmarks(
        self, landmarks, width: int, height: int
    ) -> Optional[Tuple[float, float, float]]:
        """Run solvePnP on the 6 head-pose landmarks and return (pitch, yaw, roll)."""
        image_points = np.array(
            [
                (landmarks[NOSE_TIP_IDX].x * width, landmarks[NOSE_TIP_IDX].y * height),
                (landmarks[CHIN_IDX].x * width, landmarks[CHIN_IDX].y * height),
                (landmarks[LEFT_EYE_CORNER_IDX].x * width, landmarks[LEFT_EYE_CORNER_IDX].y * height),
                (landmarks[RIGHT_EYE_CORNER_IDX].x * width, landmarks[RIGHT_EYE_CORNER_IDX].y * height),
                (landmarks[LEFT_MOUTH_CORNER_IDX].x * width, landmarks[LEFT_MOUTH_CORNER_IDX].y * height),
                (landmarks[RIGHT_MOUTH_CORNER_IDX].x * width, landmarks[RIGHT_MOUTH_CORNER_IDX].y * height),
            ],
            dtype=np.float64,
        )

        # Approximate camera matrix (no calibration available) — see module docstring.
        focal_length = width
        center = (width / 2.0, height / 2.0)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1))  # assume no lens distortion

        success, rotation_vector, _ = cv2.solvePnP(
            _MODEL_POINTS_3D,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return None

        return _rotation_vector_to_euler_angles(rotation_vector)

    def detect(self, frame: np.ndarray) -> AttentionResult:
        """
        Estimate attention state for the current frame (standalone use —
        runs its own FaceMesh inference). For shared-inference use, see
        analyze_landmarks().

        Args:
            frame: Input frame (BGR)

        Returns:
            AttentionResult with attention classification and pose angles.
            Returns UNKNOWN if no face is found or pose estimation fails —
            callers should NOT treat UNKNOWN as either attentive or
            distracted; it means "no reliable signal this frame."
        """
        if self._face_mesh is None:
            return AttentionResult(attention=ATTENTION_UNKNOWN)

        try:
            height, width = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self._face_mesh.process(rgb)

            if not results.multi_face_landmarks:
                return AttentionResult(attention=ATTENTION_UNKNOWN)

            landmarks = results.multi_face_landmarks[0].landmark
            return self.analyze_landmarks(landmarks, width, height)

        except Exception as e:
            logger.error(f"AttentionDetector: error during detection: {e}", exc_info=True)
            return AttentionResult(attention=ATTENTION_UNKNOWN)

    def analyze_landmarks(self, landmarks, width: int, height: int) -> AttentionResult:
        """
        Compute attention state from already-extracted MediaPipe face landmarks.

        Use this directly (instead of detect()) when landmarks were already
        computed elsewhere in the same frame (e.g. by DriverAnalyzer sharing
        one FaceMesh inference across EyeDetector and AttentionDetector).

        Args:
            landmarks: MediaPipe NormalizedLandmarkList.landmark (indexable)
            width: Frame width in pixels
            height: Frame height in pixels

        Returns:
            AttentionResult with attention classification and pose angles
        """
        try:
            pose = self._estimate_pose_from_landmarks(landmarks, width, height)
            if pose is None:
                return AttentionResult(attention=ATTENTION_UNKNOWN)

            pitch, yaw, roll = pose
            is_attentive = abs(yaw) <= self.yaw_threshold and abs(pitch) <= self.pitch_threshold
            attention = ATTENTION_ATTENTIVE if is_attentive else ATTENTION_DISTRACTED

            return AttentionResult(attention=attention, yaw=yaw, pitch=pitch, roll=roll)
        except Exception as e:
            logger.error(f"AttentionDetector: error analyzing landmarks: {e}", exc_info=True)
            return AttentionResult(attention=ATTENTION_UNKNOWN)

    def draw_attention(self, frame: np.ndarray, result: AttentionResult) -> np.ndarray:
        """Draw attention status and pose angles on frame."""
        annotated = frame.copy()
        if result.attention != ATTENTION_UNKNOWN:
            color = (0, 200, 0) if result.attention == ATTENTION_ATTENTIVE else (0, 0, 255)
            label = f"Attention: {result.attention} (yaw {result.yaw:.0f}, pitch {result.pitch:.0f})"
            cv2.putText(
                annotated, label, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )
        return annotated

    def is_available(self) -> bool:
        """Whether the underlying model loaded successfully."""
        return self._face_mesh is not None
