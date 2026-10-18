"""
Road Hazard Detector.

IMPORTANT — read before modifying:
    There is no widely-available, verifiable general-purpose pretrained
    model for pothole/crack/debris detection bundled with standard CV
    libraries the way YOLO ships for common objects (car, person, etc.).
    Third-party "pothole detection" weights exist on model hubs, but this
    project does NOT auto-download or silently trust an unverified external
    model, per the project's explicit rule against inventing or assuming
    pretrained models that cannot be confirmed to work.

    Instead, this module implements a PLUGGABLE interface:

    1. PRETRAINED MODE (preferred, opt-in):
       If the user places a real object-detection model (any Ultralytics-
       compatible .pt file trained on hazard classes such as pothole/crack/
       debris) at the path given by `HAZARD_MODEL_PATH`, this module loads
       and uses it exactly like VehicleDetector uses YOLO. Detections in
       this mode are reported with source="pretrained_model".

    2. HEURISTIC FALLBACK MODE (default, clearly labeled):
       If no hazard model is configured, this module falls back to a
       simple OpenCV texture/contrast anomaly heuristic that flags
       unusually dark, irregular blobs within the road region of interest.
       This is NOT a trained hazard classifier. It is a best-effort visual
       heuristic that will produce false positives/negatives and should be
       treated as experimental. Every result from this path is tagged
       source="heuristic_fallback" so downstream consumers (risk engine,
       dashboard, LLM) never confuse it with a validated model output.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional
import cv2
import numpy as np

from src.utils.config import Config
from src.utils.logger import logger


@dataclass
class HazardDetection:
    """A single detected road hazard."""

    hazard_type: str
    confidence: float
    bbox: Optional[tuple] = None
    source: str = "heuristic_fallback"  # "pretrained_model" or "heuristic_fallback"


@dataclass
class HazardResult:
    """Result of hazard detection for a frame."""

    hazard_detected: bool
    hazards: List[HazardDetection] = field(default_factory=list)
    mode: str = "heuristic_fallback"

    def to_dict(self) -> dict:
        """Convert to the plain dict shape described in the project spec."""
        return {
            "hazard_detected": self.hazard_detected,
            "hazards": [
                {"type": h.hazard_type, "confidence": round(h.confidence, 3)}
                for h in self.hazards
            ],
        }


class HazardDetector:
    """
    Detects road hazards (pothole, crack, obstacle, debris).

    Operates in one of two modes:
        - "pretrained_model": a real hazard-detection model was configured and loaded
        - "heuristic_fallback": no model configured; uses an OpenCV heuristic instead

    The active mode is always exposed via `self.mode` and included in results,
    so the rest of the system never mistakes heuristic output for validated
    model predictions.
    """

    HAZARD_CLASSES = ["pothole", "crack", "obstacle", "debris"]

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.40,
        heuristic_min_area: int = 800,
        heuristic_max_area: int = 15000,
        heuristic_darkness_threshold: int = 40,
        heuristic_background_kernel: int = 51,
        roi_top_ratio: float = 0.55,
        roi_bottom_ratio: float = 1.0,
    ) -> None:
        """
        Initialize hazard detector.

        Args:
            model_path: Path to a pretrained hazard-detection model (.pt).
                If None or the file doesn't exist, falls back to heuristic mode.
            confidence_threshold: Minimum confidence for pretrained-model detections
            heuristic_min_area: Min contour area (px) for heuristic candidate regions
            heuristic_max_area: Max contour area (px) for heuristic candidate regions
            heuristic_darkness_threshold: Intensity difference (0-255) below which a
                region is NOT considered anomalous relative to the local road surface
            heuristic_background_kernel: Box-blur kernel size (px) used to estimate
                the local road-surface background. Must be larger than the expected
                hazard feature size, or the feature will bleed into its own
                "background" estimate and go undetected.
            roi_top_ratio: Fraction of frame height above which is ignored (sky/horizon)
            roi_bottom_ratio: Fraction of frame height BELOW which is ignored. Default
                1.0 assumes the road is visible to the bottom edge of the frame. If your
                camera is mounted low enough that the hood/dashboard is visible in the
                lower part of the frame, LOWER this value so the ROI stops before the
                hood — otherwise hood reflections get analyzed as "road anomalies."
        """
        self.model_path = model_path or os.getenv("HAZARD_MODEL_PATH", "")
        self.confidence_threshold = confidence_threshold
        self.heuristic_min_area = heuristic_min_area
        self.heuristic_max_area = heuristic_max_area
        self.heuristic_darkness_threshold = heuristic_darkness_threshold
        # Box-blur kernel must be odd for symmetric filtering
        self.heuristic_background_kernel = (
            heuristic_background_kernel
            if heuristic_background_kernel % 2 == 1
            else heuristic_background_kernel + 1
        )
        self.roi_top_ratio = roi_top_ratio
        self.roi_bottom_ratio = roi_bottom_ratio

        self.model = None
        self.mode = "heuristic_fallback"
        self.class_names = None

        self._load()

    def _load(self) -> None:
        """Attempt to load a configured pretrained hazard model; else stay in fallback mode."""
        if not self.model_path:
            logger.warning(
                "HazardDetector: no HAZARD_MODEL_PATH configured. "
                "Using OpenCV heuristic fallback (experimental, not a trained model)."
            )
            self.mode = "heuristic_fallback"
            return

        if not os.path.exists(self.model_path):
            logger.warning(
                f"HazardDetector: configured model path '{self.model_path}' does not exist. "
                "Using OpenCV heuristic fallback (experimental, not a trained model)."
            )
            self.mode = "heuristic_fallback"
            return

        try:
            from ultralytics import YOLO

            self.model = YOLO(self.model_path)
            self.class_names = self.model.names
            self.mode = "pretrained_model"
            logger.info(f"HazardDetector: loaded pretrained model from {self.model_path}")
            logger.info(f"HazardDetector: model classes: {list(self.class_names.values())}")
        except Exception as e:
            logger.error(f"HazardDetector: failed to load model at {self.model_path}: {e}")
            logger.warning("HazardDetector: falling back to OpenCV heuristic.")
            self.model = None
            self.mode = "heuristic_fallback"

    def _road_roi_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Restrict analysis to the road surface, excluding sky above
        roi_top_ratio and anything below roi_bottom_ratio (e.g. a visible
        hood on hood/dashboard-mounted cameras).
        """
        height, width = frame.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        top_y = int(height * self.roi_top_ratio)
        bottom_y = int(height * self.roi_bottom_ratio)
        polygon = np.array(
            [
                [
                    (int(width * 0.10), bottom_y),
                    (int(width * 0.35), top_y),
                    (int(width * 0.65), top_y),
                    (int(width * 0.90), bottom_y),
                ]
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, polygon, 255)
        return mask

    def _detect_pretrained(self, frame: np.ndarray) -> List[HazardDetection]:
        """Run inference using the configured pretrained hazard model."""
        detections = []
        try:
            results = self.model(frame, conf=self.confidence_threshold, verbose=False)
            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.class_names.get(class_id, "hazard")

                    detections.append(
                        HazardDetection(
                            hazard_type=class_name,
                            confidence=confidence,
                            bbox=(float(x1), float(y1), float(x2), float(y2)),
                            source="pretrained_model",
                        )
                    )
        except Exception as e:
            logger.error(f"HazardDetector: inference error: {e}", exc_info=True)

        return detections

    def _detect_heuristic(self, frame: np.ndarray) -> List[HazardDetection]:
        """
        Experimental OpenCV heuristic: flags irregular dark blobs within the
        road ROI as *candidate* hazards (e.g. potholes/shadow patches).

        This intentionally does NOT try to classify hazard type with
        confidence — it labels findings as generic "obstacle" candidates
        with modest, capped confidence, to avoid implying model-grade
        certainty for what is really a contrast heuristic.
        """
        detections = []
        try:
            roi_mask = self._road_roi_mask(frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

            # Estimate the local road-surface background using a wide box
            # blur, then flag pixels notably darker than that local
            # background (a rough proxy for potholes/cracks/debris shadows).
            # The blur kernel must be wider than the hazard itself, or the
            # feature contaminates its own background estimate and the
            # diff washes out.
            k = self.heuristic_background_kernel
            background = cv2.blur(gray, (k, k))
            diff = np.clip(background - gray, 0, 255).astype(np.uint8)
            diff = cv2.bitwise_and(diff, diff, mask=roi_mask)

            _, thresh = cv2.threshold(
                diff, self.heuristic_darkness_threshold, 255, cv2.THRESH_BINARY
            )
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

            contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                area = cv2.contourArea(contour)
                if self.heuristic_min_area <= area <= self.heuristic_max_area:
                    x, y, w, h = cv2.boundingRect(contour)

                    # Confidence is deliberately capped low/mid range: this
                    # heuristic cannot reliably discriminate hazard type or
                    # severity, only "locally anomalous dark region."
                    intensity_score = float(np.mean(diff[y : y + h, x : x + w])) / 255.0
                    confidence = min(0.35 + intensity_score * 0.3, 0.65)

                    detections.append(
                        HazardDetection(
                            hazard_type="obstacle",
                            confidence=round(confidence, 3),
                            bbox=(float(x), float(y), float(x + w), float(y + h)),
                            source="heuristic_fallback",
                        )
                    )
        except Exception as e:
            logger.error(f"HazardDetector: heuristic error: {e}", exc_info=True)

        return detections

    def detect(self, frame: np.ndarray) -> HazardResult:
        """
        Detect road hazards in a frame.

        Args:
            frame: Input frame (BGR)

        Returns:
            HazardResult with hazard_detected flag and list of hazards.
            `mode` indicates whether results came from a real pretrained
            model or the heuristic fallback — always check this before
            treating results as high-confidence AI predictions.
        """
        if self.mode == "pretrained_model" and self.model is not None:
            hazards = self._detect_pretrained(frame)
        else:
            hazards = self._detect_heuristic(frame)

        return HazardResult(
            hazard_detected=len(hazards) > 0,
            hazards=hazards,
            mode=self.mode,
        )

    def draw_hazards(self, frame: np.ndarray, result: HazardResult) -> np.ndarray:
        """Draw hazard bounding boxes on frame, color-coded by source mode."""
        annotated = frame.copy()

        color = (0, 140, 255) if result.mode == "pretrained_model" else (0, 200, 200)

        for hazard in result.hazards:
            if hazard.bbox is None:
                continue
            x1, y1, x2, y2 = [int(v) for v in hazard.bbox]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            tag = "" if hazard.source == "pretrained_model" else " (heuristic)"
            label = f"{hazard.hazard_type} {hazard.confidence:.2f}{tag}"
            cv2.putText(
                annotated,
                label,
                (x1, max(y1 - 8, 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

        return annotated

    def is_using_pretrained_model(self) -> bool:
        """Whether a real pretrained hazard model is currently active."""
        return self.mode == "pretrained_model"
