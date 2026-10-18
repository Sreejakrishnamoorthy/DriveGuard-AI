"""
Vehicle Detector for Road Analysis.

Uses YOLO to detect vehicles, pedestrians, and bicycles in road camera footage.
"""

from dataclasses import dataclass
from typing import List, Tuple
import cv2
import numpy as np

from src.models.model_manager import model_manager
from src.utils.config import Config
from src.utils.logger import logger


@dataclass
class Detection:
    """Data class for object detection result."""

    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    class_id: int

    def center(self) -> Tuple[float, float]:
        """Get bounding box center point."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def width(self) -> float:
        """Get bounding box width."""
        x1, y1, x2, y2 = self.bbox
        return x2 - x1

    def height(self) -> float:
        """Get bounding box height."""
        x1, y1, x2, y2 = self.bbox
        return y2 - y1

    def area(self) -> float:
        """Get bounding box area."""
        return self.width() * self.height()


class VehicleDetector:
    """
    Detects vehicles, pedestrians, and bicycles in road camera frames.

    Uses pretrained YOLO model to detect:
    - car
    - truck
    - bus
    - motorcycle
    - bicycle
    - person (pedestrian)
    """

    # Classes to detect in road camera
    ROAD_CLASSES = {
        "car": 2,
        "truck": 7,
        "bus": 5,
        "motorcycle": 3,
        "bicycle": 1,
        "person": 0,
    }

    def __init__(self, confidence_threshold: float = 0.40) -> None:
        """
        Initialize vehicle detector.

        Args:
            confidence_threshold: Minimum confidence for detections
        """
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.class_names = None

        logger.info(f"Initializing VehicleDetector (threshold: {confidence_threshold})")

    def load_model(self) -> None:
        """Load YOLO model from model manager."""
        if self.model is not None:
            logger.debug("Model already loaded.")
            return

        try:
            self.model = model_manager.get_yolo_model()
            self.class_names = self.model.names

            logger.info("VehicleDetector model loaded successfully.")
            logger.debug(f"Available classes: {list(self.class_names.values())}")

        except Exception as e:
            logger.error(f"Failed to load VehicleDetector model: {e}")
            raise

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect objects in a frame.

        Args:
            frame: Input frame (numpy array, BGR format)

        Returns:
            List of Detection objects
        """
        if self.model is None:
            self.load_model()

        try:
            # Run inference
            results = self.model(
                frame,
                conf=self.confidence_threshold,
                verbose=False,
            )

            detections = []

            # Process results
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        # Extract box information
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())

                        # Get class name
                        class_name = self.class_names.get(class_id, "unknown")

                        # Create detection
                        detection = Detection(
                            class_name=class_name,
                            confidence=confidence,
                            bbox=(float(x1), float(y1), float(x2), float(y2)),
                            class_id=class_id,
                        )

                        detections.append(detection)

            return detections

        except Exception as e:
            logger.error(f"Error during vehicle detection: {e}", exc_info=True)
            return []

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """
        Draw detection bounding boxes on frame.

        Args:
            frame: Input frame
            detections: List of detections
            color: Box color (BGR)
            thickness: Box line thickness

        Returns:
            Annotated frame
        """
        annotated = frame.copy()

        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            # Draw label
            label = f"{detection.class_name} {detection.confidence:.2f}"
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
                (255, 255, 255),
                1,
            )

        return annotated

    def get_detection_summary(self, detections: List[Detection]) -> dict:
        """
        Get a summary of detections.

        Args:
            detections: List of detections

        Returns:
            Dictionary with detection counts
        """
        summary = {
            "total_detections": len(detections),
            "class_counts": {},
            "high_confidence": 0,
        }

        for detection in detections:
            # Count by class
            if detection.class_name not in summary["class_counts"]:
                summary["class_counts"][detection.class_name] = 0
            summary["class_counts"][detection.class_name] += 1

            # Count high confidence detections
            if detection.confidence >= 0.8:
                summary["high_confidence"] += 1

        return summary
