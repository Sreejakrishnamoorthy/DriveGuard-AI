"""
Analysis Service.

Central orchestration layer for the FastAPI backend. Owns the long-lived
analyzer/engine instances (so models load once, not per-request) and the
application's "current state" that GET endpoints read from.

This is the only place that wires RoadAnalyzer + DriverAnalyzer +
PhoneDetector + RiskCalculator + AlertManager + ConfidenceScorer +
IncidentRecorder + ExplanationService + DatabaseService together — route
handlers stay thin and just call into this service.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from src.road_analysis.road_analyzer import RoadAnalyzer, RoadAnalysisResult
from src.driver_monitoring.driver_analyzer import DriverAnalyzer, DriverAnalysisResult
from src.driver_monitoring.phone_detector import PhoneDetector
from src.risk_engine.risk_calculator import RiskCalculator, RiskFactors, RiskScoreResult
from src.risk_engine.alert_manager import AlertManager, Alert
from src.risk_engine.confidence_scorer import compute_confidence, ConfidenceResult
from src.risk_engine.incident_recorder import IncidentRecorder
from src.llm.explanation_service import ExplanationService, RiskContext
from backend.services.database_service import DatabaseService
from src.utils.config import Config
from src.utils.logger import logger


@dataclass
class CurrentState:
    """In-memory snapshot of the most recent analysis, for GET endpoints."""

    road: Optional[RoadAnalysisResult] = None
    driver: Optional[DriverAnalysisResult] = None
    risk: Optional[RiskScoreResult] = None
    confidence: Optional[ConfidenceResult] = None
    phone_detected: bool = False
    explanation: Optional[str] = None
    recent_alerts: List[Alert] = field(default_factory=list)
    last_updated: Optional[float] = None
    active_session_id: Optional[str] = None


class AnalysisService:
    """
    Orchestrates the full DriveGuard AI pipeline for the FastAPI backend.

    One instance is created at application startup and reused across
    requests (see backend/main.py), so YOLO/MediaPipe models are loaded
    exactly once regardless of how many requests arrive.
    """

    def __init__(self) -> None:
        logger.info("Initializing AnalysisService...")

        self.road_analyzer = RoadAnalyzer(confidence_threshold=Config.CONFIDENCE_THRESHOLD)
        self.driver_analyzer = DriverAnalyzer()
        self.phone_detector = PhoneDetector()
        self.risk_calculator = RiskCalculator()
        self.alert_manager = AlertManager()
        self.explanation_service = ExplanationService()
        self.database_service = DatabaseService()
        self.incident_recorder = IncidentRecorder()

        self.state = CurrentState()

        logger.info("AnalysisService initialized successfully.")
        logger.info(f"Database connected: {self.database_service.is_connected}")

    @staticmethod
    def _derive_risk_factors(
        driver_result: Optional[DriverAnalysisResult],
        road_result: Optional[RoadAnalysisResult],
        phone_detected: bool = False,
    ) -> RiskFactors:
        """
        Build RiskFactors from whatever analysis results are currently
        available. Phone detection (Phase 25) is folded into the existing
        `distraction` factor via OR — a driver holding a phone is a form
        of distraction — rather than added as a new weighted category,
        which would change the exact scoring formula the project spec
        defines (30/25/20/15/20 points, capped at 100).
        """
        drowsiness = bool(driver_result and driver_result.drowsiness)
        head_pose_distracted = bool(driver_result and driver_result.attention == "DISTRACTED")
        distraction = head_pose_distracted or phone_detected

        lane_violation = bool(
            road_result and road_result.lane and road_result.lane.lane_departure
        )
        road_hazard = bool(
            road_result and road_result.hazards and road_result.hazards.hazard_detected
        )
        unsafe_distance = bool(road_result and road_result.following_distance == "UNSAFE")

        return RiskFactors(
            drowsiness=drowsiness,
            unsafe_following_distance=unsafe_distance,
            lane_violation=lane_violation,
            road_hazard=road_hazard,
            distraction=distraction,
        )

    def analyze_frame(
        self,
        road_frame: Optional[np.ndarray] = None,
        driver_frame: Optional[np.ndarray] = None,
        timestamp: Optional[float] = None,
        generate_explanation: bool = True,
    ) -> Dict[str, Any]:
        """
        Run one analysis cycle: road analysis, driver analysis, phone
        detection, risk scoring, confidence scoring, alert generation,
        and (for high-severity alerts) incident recording. Updates
        self.state and returns a combined result dict.

        Args:
            road_frame: Road camera frame (BGR), or None to skip road analysis
            driver_frame: Driver camera frame (BGR), or None to skip driver analysis
            timestamp: Explicit timestamp (seconds); defaults to wall-clock time
            generate_explanation: Whether to also call the LLM explanation
                service this cycle (skip for high-frequency frame loops and
                only call periodically, since it's the slowest step)

        Returns:
            Combined dict with road/driver/risk/confidence/explanation/alerts
        """
        if timestamp is None:
            timestamp = time.time()

        road_result = (
            self.road_analyzer.analyze(road_frame, timestamp=timestamp)
            if road_frame is not None
            else None
        )
        driver_result = (
            self.driver_analyzer.analyze(driver_frame, timestamp=timestamp)
            if driver_frame is not None
            else None
        )

        phone_result = self.phone_detector.detect(driver_frame) if driver_frame is not None else None
        phone_detected = bool(phone_result and phone_result.phone_detected)

        factors = self._derive_risk_factors(driver_result, road_result, phone_detected)
        risk_result = self.risk_calculator.calculate(factors)

        near_miss = road_result.near_miss if road_result else None
        new_alerts = self.alert_manager.process(
            factors,
            risk_result,
            timestamp=timestamp,
            near_miss_level=near_miss.near_miss_level if near_miss else None,
            near_miss_ttc_seconds=near_miss.time_to_collision_seconds if near_miss else None,
        )

        confidence_result = compute_confidence(road_result=road_result, driver_result=driver_result)

        explanation_text = self.state.explanation
        if generate_explanation:
            context = RiskContext(
                risk_score=risk_result.risk_score,
                risk_level=risk_result.risk_level,
                drowsiness=factors.drowsiness,
                distraction=factors.distraction,
                lane_violation=factors.lane_violation,
                unsafe_following_distance=factors.unsafe_following_distance,
                road_hazard=factors.road_hazard,
            )
            explanation_text = self.explanation_service.generate_explanation(context).text

        self.state.road = road_result
        self.state.driver = driver_result
        self.state.risk = risk_result
        self.state.confidence = confidence_result
        self.state.phone_detected = phone_detected
        self.state.explanation = explanation_text
        self.state.last_updated = timestamp
        if new_alerts:
            self.state.recent_alerts = self.alert_manager.get_prioritized_alerts(limit=50)

        # Phase 27 — Incident Reconstruction: auto-record a snapshot for
        # any HIGH/CRITICAL alert that just fired.
        for alert in new_alerts:
            if IncidentRecorder.should_record(alert.severity):
                self.incident_recorder.record(
                    alert_type=alert.type,
                    severity=alert.severity,
                    message=alert.message,
                    risk_score=risk_result.risk_score,
                    risk_level=risk_result.risk_level,
                    factor_breakdown=risk_result.factor_breakdown,
                    road_status=road_result.to_dict() if road_result else None,
                    driver_status=driver_result.to_dict() if driver_result else None,
                    confidence=confidence_result.to_dict(),
                    road_frame=(
                        self.road_analyzer.draw_annotations(road_frame.copy(), road_result)
                        if road_result is not None
                        else None
                    ),
                    driver_frame=(
                        self.driver_analyzer.draw_annotations(driver_frame.copy(), driver_result)
                        if driver_result is not None
                        else None
                    ),
                    timestamp=timestamp,
                )

        if self.database_service.is_connected and self.state.active_session_id:
            self.database_service.insert_risk_event(
                {"risk_score": risk_result.risk_score, "risk_level": risk_result.risk_level},
                session_id=self.state.active_session_id,
            )
            for alert in new_alerts:
                self.database_service.insert_alert(alert.to_dict(), session_id=self.state.active_session_id)

        return {
            "road": road_result.to_dict() if road_result else None,
            "driver": driver_result.to_dict() if driver_result else None,
            "phone_detected": phone_detected,
            "risk_score": risk_result.risk_score,
            "risk_level": risk_result.risk_level,
            "risk_explanation": risk_result.explain(),
            "confidence": confidence_result.to_dict(),
            "explanation": explanation_text,
            "new_alerts": [a.to_dict() for a in new_alerts],
        }

    def get_current_risk(self) -> Dict[str, Any]:
        """Get the most recent risk score/level/explanation."""
        if self.state.risk is None:
            return {"risk_score": 0, "risk_level": "LOW", "explanation": None}
        return {
            "risk_score": self.state.risk.risk_score,
            "risk_level": self.state.risk.risk_level,
            "explanation": self.state.explanation,
        }

    def get_risk_explanation(self) -> Dict[str, Any]:
        """Phase 28 — get the full explainable-risk breakdown for the latest assessment."""
        if self.state.risk is None:
            return {
                "risk_score": 0,
                "risk_level": "LOW",
                "raw_score_before_cap": 0,
                "was_capped": False,
                "factor_breakdown": [],
            }
        return self.state.risk.explain()

    def get_confidence(self) -> Dict[str, Any]:
        """Phase 26 — get the confidence assessment for the latest analysis."""
        if self.state.confidence is None:
            return {"overall_confidence": None, "confidence_label": "UNKNOWN", "components": []}
        return self.state.confidence.to_dict()

    def get_driver_status(self) -> Dict[str, Any]:
        """Get the most recent driver analysis, or a default 'no data yet' shape."""
        if self.state.driver is None:
            return {
                "face_detected": False,
                "eye_state": "UNKNOWN",
                "drowsiness": False,
                "drowsiness_probability": 0.0,
                "attention": "UNKNOWN",
                "driver_status": "NO_DATA",
            }
        result = self.state.driver.to_dict()
        result["phone_detected"] = self.state.phone_detected
        return result

    def get_road_status(self) -> Dict[str, Any]:
        """Get the most recent road analysis, or a default 'no data yet' shape."""
        if self.state.road is None:
            return {
                "vehicles": [],
                "lane": {"lane_detected": False, "lane_status": "UNKNOWN", "lane_departure": False},
                "hazards": [],
                "hazard_detected": False,
                "following_distance": "SAFE",
                "vehicle_summary": {},
                "near_miss": {"near_miss_level": "NONE", "time_to_collision_seconds": None},
            }
        return self.state.road.to_dict()

    def get_alerts(self, limit: int = 50, prioritized: bool = True) -> List[Dict[str, Any]]:
        """
        Get recent alerts. Phase 20 — by default, ordered by severity
        (most urgent first) rather than plain recency.
        """
        if prioritized:
            return [a.to_dict() for a in self.alert_manager.get_prioritized_alerts(limit=limit)]
        return [a.to_dict() for a in self.alert_manager.get_recent_alerts(limit=limit)]

    def get_incidents(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Phase 27 — list recorded incidents, most recent first."""
        return self.incident_recorder.list_incidents(limit=limit)

    def get_incident(self, incident_id: str) -> Optional[Dict[str, Any]]:
        """Phase 27 — get a single incident's full record by ID."""
        return self.incident_recorder.get_incident(incident_id)

    def reset(self) -> None:
        """Reset all temporal/analyzer state (e.g. starting a fresh video)."""
        self.road_analyzer.reset()
        self.driver_analyzer.reset()
        self.alert_manager.reset()
        self.state = CurrentState()
        logger.info("AnalysisService: full state reset.")
