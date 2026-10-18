"""
Phone Detector — Phase 25.

Detects a cell phone in the driver-facing camera frame, as an additional
distraction signal alongside head-pose-based attention. Uses the SAME
shared YOLO model already loaded for road-camera vehicle detection (via
ModelManager) — YOLOv8's standard COCO weights already include a "cell
phone" class, so this is a real, genuinely pretrained capability being
reused, not a new model or a fabricated heuristic.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from src.models.model_manager import model_manager
from src.utils.logger import logger

CELL_PHONE_CLASS_NAME = "cell phone"


@dataclass
class PhoneDetectionResult:
    """Result of phone detection for a single frame."""

    phone_detected: bool
    confidence: float = 0.0
    bbox: Optional[Tuple[float, float, float, float]] = None

    def to_dict(self) -> dict:
        return {
            "phone_detected": self.phone_detected,
            "confidence": round(self.confidence, 3) if self.phone_detected else 0.0,
        }


class PhoneDetector:
    """
    Detects phone use in the driver-facing camera using the shared YOLO
    model's built-in "cell phone" COCO class.
    """

    def __init__(self, confidence_threshold: float = 0.35) -> None:
        """
        Initialize the phone detector.

        Args:
            confidence_threshold: Minimum YOLO confidence to report a
                phone detection. Kept slightly lower than the main vehicle
                detector's default since phones are small objects that
                tend to get lower confidence scores even when correctly
                detected.
        """
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.class_names = None
        self._phone_class_id: Optional[int] = None

    def load_model(self) -> None:
        """Load (or reuse) the shared YOLO model from ModelManager."""
        if self.model is not None:
            return

        self.model = model_manager.get_yolo_model()
        if self.model is None:
            logger.warning("PhoneDetector: shared YOLO model unavailable.")
            return

        self.class_names = self.model.names
        matches = [cid for cid, name in self.class_names.items() if name == CELL_PHONE_CLASS_NAME]
        if not matches:
            logger.warning(
                f"PhoneDetector: '{CELL_PHONE_CLASS_NAME}' class not found in loaded model's "
                f"class list — phone detection will be unavailable with this model."
            )
            return
        self._phone_class_id = matches[0]
        logger.info(f"PhoneDetector: using shared YOLO model, phone class id={self._phone_class_id}")

    def is_available(self) -> bool:
        """Whether phone detection can actually run with the currently loaded model."""
        return self.model is not None and self._phone_class_id is not None

    def detect(self, frame: np.ndarray) -> PhoneDetectionResult:
        """
        Detect a phone in the driver-facing frame.

        Args:
            frame: Input frame (BGR), typically from the driver-facing camera

        Returns:
            PhoneDetectionResult. Returns phone_detected=False (not an
            error) if the model/class is unavailable, so callers can
            treat this as "no signal" safely.
        """
        if self.model is None:
            self.load_model()

        if not self.is_available():
            return PhoneDetectionResult(phone_detected=False)

        try:
            results = self.model(frame, conf=self.confidence_threshold, verbose=False)

            best_conf = 0.0
            best_bbox = None

            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    class_id = int(box.cls[0].cpu().numpy())
                    if class_id != self._phone_class_id:
                        continue
                    confidence = float(box.conf[0].cpu().numpy())
                    if confidence > best_conf:
                        best_conf = confidence
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        best_bbox = (float(x1), float(y1), float(x2), float(y2))

            if best_bbox is not None:
                return PhoneDetectionResult(phone_detected=True, confidence=best_conf, bbox=best_bbox)
            return PhoneDetectionResult(phone_detected=False)

        except Exception as e:
            logger.error(f"PhoneDetector: error during detection: {e}", exc_info=True)
            return PhoneDetectionResult(phone_detected=False)

    def draw_detection(self, frame: np.ndarray, result: PhoneDetectionResult) -> np.ndarray:
        """Draw a phone-detection bounding box on the frame, if one was found."""
        if not result.phone_detected or result.bbox is None:
            return frame
        annotated = frame.copy()
        x1, y1, x2, y2 = [int(v) for v in result.bbox]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            annotated,
            f"phone {result.confidence:.2f}",
            (x1, max(y1 - 8, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )
        return annotated
