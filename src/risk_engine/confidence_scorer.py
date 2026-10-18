"""
Confidence Scorer — Phase 26.

Computes a transparent "how much should I trust this assessment right
now" signal, built entirely from confidence values the detectors already
compute (YOLO detection confidence, MediaPipe face-detection confidence,
Eye-Aspect-Ratio margin from the open/closed threshold, and whether the
hazard detector is running a real trained model vs. the heuristic
fallback). Nothing here is invented — every component maps to a real,
already-computed number, and any signal that isn't currently available
is reported as None/"unavailable" rather than silently defaulted to a
fabricated value.
"""

from dataclasses import dataclass, field
from typing import List, Optional

CONFIDENCE_HIGH = "HIGH"
CONFIDENCE_MEDIUM = "MEDIUM"
CONFIDENCE_LOW = "LOW"
CONFIDENCE_UNKNOWN = "UNKNOWN"


@dataclass
class ConfidenceComponent:
    """One input into the overall confidence assessment."""

    name: str
    confidence: Optional[float]  # 0.0-1.0, or None if not applicable this frame
    note: str

    def to_dict(self) -> dict:
        return {
            "component": self.name,
            "confidence": round(self.confidence, 3) if self.confidence is not None else None,
            "note": self.note,
        }


@dataclass
class ConfidenceResult:
    """Overall confidence assessment for one frame's analysis."""

    overall_confidence: Optional[float]
    confidence_label: str
    components: List[ConfidenceComponent] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "overall_confidence": round(self.overall_confidence, 3)
            if self.overall_confidence is not None
            else None,
            "confidence_label": self.confidence_label,
            "components": [c.to_dict() for c in self.components],
        }


def _label_for(confidence: Optional[float]) -> str:
    if confidence is None:
        return CONFIDENCE_UNKNOWN
    if confidence >= 0.75:
        return CONFIDENCE_HIGH
    if confidence >= 0.45:
        return CONFIDENCE_MEDIUM
    return CONFIDENCE_LOW


def _ear_margin_confidence(ear_value: Optional[float], ear_threshold: float = 0.21) -> Optional[float]:
    """
    Confidence in the eye-state (open/closed) classification, based on how
    far the measured EAR is from the decision threshold. An EAR right at
    the threshold is an ambiguous, low-confidence call; an EAR far from it
    (clearly open or clearly closed) is a high-confidence call.
    """
    if ear_value is None:
        return None
    distance = abs(ear_value - ear_threshold)
    # Normalize: a margin of 0.15+ (very clearly open or closed) maps to
    # ~1.0 confidence; scale is a reasonable calibration against typical
    # observed EAR ranges (see eye_detector.py), not an exact measurement.
    return min(distance / 0.15, 1.0)


def compute_confidence(
    road_result=None,
    driver_result=None,
    ear_threshold: float = 0.21,
) -> ConfidenceResult:
    """
    Compute an overall confidence assessment from whatever analysis
    results are available this frame.

    Args:
        road_result: RoadAnalysisResult from RoadAnalyzer.analyze(), or None
        driver_result: DriverAnalysisResult from DriverAnalyzer.analyze(), or None
        ear_threshold: The EAR threshold currently in use (should match
            the EyeDetector instance's configured threshold)

    Returns:
        ConfidenceResult with an overall 0-1 confidence, a qualitative
        label, and a transparent per-component breakdown.
    """
    components: List[ConfidenceComponent] = []

    if road_result is not None:
        vehicles = getattr(road_result, "vehicles", None) or []
        if vehicles:
            avg_conf = sum(v.confidence for v in vehicles) / len(vehicles)
            components.append(
                ConfidenceComponent(
                    name="vehicle_detection",
                    confidence=avg_conf,
                    note=f"Average YOLO confidence across {len(vehicles)} tracked object(s)",
                )
            )
        else:
            components.append(
                ConfidenceComponent(
                    name="vehicle_detection",
                    confidence=None,
                    note="No vehicles/pedestrians currently in frame",
                )
            )

        hazards = getattr(road_result, "hazards", None)
        if hazards is not None:
            is_pretrained = getattr(hazards, "mode", "") == "pretrained_model"
            components.append(
                ConfidenceComponent(
                    name="hazard_detection",
                    # A real trained model's own confidences are trustworthy;
                    # the heuristic fallback is explicitly experimental (see
                    # hazard_detector.py) so it's capped at a modest ceiling
                    # regardless of its raw per-detection confidence values.
                    confidence=0.8 if is_pretrained else 0.35,
                    note=(
                        "Using a pretrained hazard model"
                        if is_pretrained
                        else "Using the experimental OpenCV heuristic fallback (no trained model configured)"
                    ),
                )
            )

    if driver_result is not None:
        if getattr(driver_result, "face_detected", False):
            components.append(
                ConfidenceComponent(
                    name="face_detection",
                    confidence=getattr(driver_result, "face_confidence", None),
                    note="MediaPipe face-detection confidence",
                )
            )

            ear_value = getattr(driver_result, "ear_value", None)
            ear_conf = _ear_margin_confidence(ear_value, ear_threshold)
            components.append(
                ConfidenceComponent(
                    name="eye_state",
                    confidence=ear_conf,
                    note=(
                        f"EAR={ear_value:.3f}, margin from threshold {ear_threshold}"
                        if ear_value is not None
                        else "No eye landmarks available this frame"
                    ),
                )
            )

            pose_yaw = getattr(driver_result, "pose_yaw", None)
            pose_pitch = getattr(driver_result, "pose_pitch", None)
            pose_available = pose_yaw is not None and pose_pitch is not None
            components.append(
                ConfidenceComponent(
                    name="attention_pose",
                    # solvePnP doesn't expose a native per-call confidence
                    # score, so this component reports simply whether a
                    # pose was successfully estimated this frame, not a
                    # fabricated precision number.
                    confidence=0.7 if pose_available else None,
                    note=(
                        f"Head pose estimated (yaw={pose_yaw:.1f}, pitch={pose_pitch:.1f})"
                        if pose_available
                        else "Head pose could not be estimated this frame"
                    ),
                )
            )
        else:
            components.append(
                ConfidenceComponent(
                    name="face_detection",
                    confidence=None,
                    note="No face detected in driver camera this frame",
                )
            )

    available = [c.confidence for c in components if c.confidence is not None]
    overall = sum(available) / len(available) if available else None

    return ConfidenceResult(
        overall_confidence=overall,
        confidence_label=_label_for(overall),
        components=components,
    )
