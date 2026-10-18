"""
Risk Calculator — the core of DriveGuard AI.

Combines structured signals from road analysis and driver monitoring into
a single 0-100 risk score and a risk level (LOW/MEDIUM/HIGH/CRITICAL).

This module has NO dependency on the LLM, MongoDB, FastAPI, or any
detection model directly — it is pure, deterministic scoring logic over
already-computed boolean/categorical inputs, which makes it trivially
testable and guarantees the core risk pipeline works even if every other
subsystem (LLM, database, etc.) is unavailable.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from src.risk_engine.risk_rules import RiskWeights, get_risk_level, clamp_score
from src.utils.logger import logger


@dataclass
class RiskFactors:
    """
    Boolean risk signals for a single evaluation, already derived from
    road/driver analysis. Keeping this as a flat, simple structure (rather
    than passing raw analyzer objects into the calculator) keeps the
    scoring logic decoupled from how those signals were produced.
    """

    drowsiness: bool = False
    unsafe_following_distance: bool = False
    lane_violation: bool = False
    road_hazard: bool = False
    distraction: bool = False


@dataclass
class RiskScoreResult:
    """Result of a risk calculation."""

    risk_score: int
    risk_level: str
    contributing_factors: List[str] = field(default_factory=list)
    raw_score_before_cap: int = 0
    factor_breakdown: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to the plain dict shape described in the project spec."""
        return {"risk_score": self.risk_score, "risk_level": self.risk_level}

    def explain(self) -> dict:
        """
        Phase 28 — Explainable Risk.

        Returns a transparent breakdown of exactly which factors
        contributed how many points to the score, so "why is my risk 75?"
        has a concrete, auditable answer instead of just the final number.
        """
        return {
            "risk_score": self.risk_score,
            "risk_level": self.risk_level,
            "raw_score_before_cap": self.raw_score_before_cap,
            "was_capped": self.raw_score_before_cap > self.risk_score,
            "factor_breakdown": self.factor_breakdown,
        }


class RiskCalculator:
    """
    Computes a 0-100 driving risk score from boolean risk factors.

    Weights are configurable (see RiskWeights); the score is always
    clamped to [0, 100] regardless of how many factors are active.
    """

    def __init__(self, weights: Optional[RiskWeights] = None) -> None:
        """
        Initialize the risk calculator.

        Args:
            weights: Custom point weights per factor. Defaults to
                RiskWeights.from_config(), which reads Config.RISK_WEIGHTS
                (falling back to the spec's default point values).
        """
        self.weights = weights or RiskWeights.from_config()
        logger.info(f"RiskCalculator initialized with weights: {self.weights}")

    def calculate(self, factors: RiskFactors) -> RiskScoreResult:
        """
        Calculate risk score from a set of boolean risk factors.

        Args:
            factors: RiskFactors with which conditions are currently active

        Returns:
            RiskScoreResult with risk_score (0-100, capped), risk_level,
            and the list of factors that contributed to the score.
        """
        raw_score = 0
        contributing = []
        breakdown = []

        if factors.drowsiness:
            raw_score += self.weights.drowsiness
            contributing.append("drowsiness")
            breakdown.append(
                {
                    "factor": "drowsiness",
                    "points": self.weights.drowsiness,
                    "description": "Driver shows signs of drowsiness",
                }
            )

        if factors.unsafe_following_distance:
            raw_score += self.weights.unsafe_following_distance
            contributing.append("unsafe_following_distance")
            breakdown.append(
                {
                    "factor": "unsafe_following_distance",
                    "points": self.weights.unsafe_following_distance,
                    "description": "Following distance to the vehicle ahead is unsafe",
                }
            )

        if factors.lane_violation:
            raw_score += self.weights.lane_violation
            contributing.append("lane_violation")
            breakdown.append(
                {
                    "factor": "lane_violation",
                    "points": self.weights.lane_violation,
                    "description": "Vehicle is departing its lane",
                }
            )

        if factors.road_hazard:
            raw_score += self.weights.road_hazard
            contributing.append("road_hazard")
            breakdown.append(
                {
                    "factor": "road_hazard",
                    "points": self.weights.road_hazard,
                    "description": "A potential road hazard was detected ahead",
                }
            )

        if factors.distraction:
            raw_score += self.weights.distraction
            contributing.append("distraction")
            breakdown.append(
                {
                    "factor": "distraction",
                    "points": self.weights.distraction,
                    "description": "Driver appears distracted from the road",
                }
            )

        capped_score = clamp_score(raw_score)
        risk_level = get_risk_level(capped_score)

        if raw_score > 100:
            logger.debug(
                f"Risk score capped: raw={raw_score} -> {capped_score} "
                f"(factors: {contributing})"
            )

        return RiskScoreResult(
            risk_score=capped_score,
            risk_level=risk_level,
            contributing_factors=contributing,
            raw_score_before_cap=raw_score,
            factor_breakdown=breakdown,
        )

    def calculate_from_analysis(self, driver_result, road_result) -> RiskScoreResult:
        """
        Convenience method: derive RiskFactors directly from a
        DriverAnalysisResult and RoadAnalysisResult and calculate risk.

        Args:
            driver_result: DriverAnalysisResult from DriverAnalyzer.analyze()
            road_result: RoadAnalysisResult from RoadAnalyzer.analyze()

        Returns:
            RiskScoreResult
        """
        lane_violation = bool(
            road_result.lane and getattr(road_result.lane, "lane_departure", False)
        )
        road_hazard = bool(
            road_result.hazards and getattr(road_result.hazards, "hazard_detected", False)
        )
        unsafe_distance = getattr(road_result, "following_distance", "SAFE") == "UNSAFE"

        factors = RiskFactors(
            drowsiness=bool(getattr(driver_result, "drowsiness", False)),
            unsafe_following_distance=unsafe_distance,
            lane_violation=lane_violation,
            road_hazard=road_hazard,
            distraction=getattr(driver_result, "attention", None) == "DISTRACTED",
        )

        return self.calculate(factors)
