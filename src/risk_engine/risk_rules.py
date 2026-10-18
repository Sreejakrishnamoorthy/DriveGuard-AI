"""
Risk Rules for the Driving Risk Engine.

Centralizes the configurable weights and risk-level thresholds used by
RiskCalculator, so scoring logic can be tuned without touching the
calculator's code.
"""

from dataclasses import dataclass

from src.utils.config import Config

RISK_LEVEL_LOW = "LOW"
RISK_LEVEL_MEDIUM = "MEDIUM"
RISK_LEVEL_HIGH = "HIGH"
RISK_LEVEL_CRITICAL = "CRITICAL"

MIN_RISK_SCORE = 0
MAX_RISK_SCORE = 100


@dataclass
class RiskWeights:
    """
    Point values added to the risk score when each factor is present.

    Defaults match the project specification exactly:
        drowsiness = 30, unsafe_following_distance = 25,
        lane_violation = 20, road_hazard = 15, distraction = 20
    """

    drowsiness: int = 30
    unsafe_following_distance: int = 25
    lane_violation: int = 20
    road_hazard: int = 15
    distraction: int = 20

    @classmethod
    def from_config(cls) -> "RiskWeights":
        """Build weights from Config.RISK_WEIGHTS, falling back to spec defaults."""
        w = Config.RISK_WEIGHTS or {}
        defaults = cls()
        return cls(
            drowsiness=int(w.get("drowsiness", defaults.drowsiness)),
            unsafe_following_distance=int(
                w.get("unsafe_following_distance", defaults.unsafe_following_distance)
            ),
            lane_violation=int(w.get("lane_violation", defaults.lane_violation)),
            road_hazard=int(w.get("road_hazard", defaults.road_hazard)),
            distraction=int(w.get("distraction", defaults.distraction)),
        )


def get_risk_level(score: float) -> str:
    """
    Classify a risk score (0-100) into a risk level.

    Thresholds match the project specification exactly:
        0-30 = LOW, 31-60 = MEDIUM, 61-80 = HIGH, 81-100 = CRITICAL

    Args:
        score: Risk score, expected in [0, 100]

    Returns:
        One of "LOW", "MEDIUM", "HIGH", "CRITICAL"
    """
    if score <= 30:
        return RISK_LEVEL_LOW
    elif score <= 60:
        return RISK_LEVEL_MEDIUM
    elif score <= 80:
        return RISK_LEVEL_HIGH
    else:
        return RISK_LEVEL_CRITICAL


def clamp_score(score: float) -> int:
    """Clamp a raw summed score into the valid [0, 100] integer range."""
    return int(max(MIN_RISK_SCORE, min(MAX_RISK_SCORE, round(score))))
