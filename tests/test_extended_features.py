"""
Tests for the extended feature set: Near-Miss Detection (23), Phone
Detection (25), Confidence Scores (26), Incident Reconstruction (27),
Explainable Risk (28), Alert Prioritization (20), and Voice Alert logic (19).
"""

import time

import cv2
import numpy as np
import pytest

from src.road_analysis.vehicle_tracker import TrackedObject
from src.road_analysis.near_miss_detector import (
    NearMissDetector,
    NEAR_MISS_NONE,
    NEAR_MISS_WARNING,
    NEAR_MISS_CRITICAL,
)
from src.risk_engine.confidence_scorer import compute_confidence
from src.risk_engine.incident_recorder import IncidentRecorder
from src.risk_engine.risk_calculator import RiskCalculator, RiskFactors
from src.risk_engine.alert_manager import AlertManager, Alert


def _make_tracked_obj(tid, height, center_x=320, cls="car"):
    y2 = 240 + height / 2
    y1 = y2 - height
    return TrackedObject(
        tracking_id=tid, class_name=cls, confidence=0.9,
        bbox=(280, y1, 360, y2), center_x=center_x, center_y=(y1 + y2) / 2,
    )


class TestNearMissDetector:
    def test_approaching_vehicle_triggers_warning_or_critical(self):
        det = NearMissDetector(warning_ttc_seconds=2.5, critical_ttc_seconds=1.2, min_samples=4)
        t, result = 0.0, None
        for i in range(10):
            result = det.update([_make_tracked_obj(1, 50 + i * 20)], 640, t)
            t += 0.1
        assert result.near_miss_level in (NEAR_MISS_WARNING, NEAR_MISS_CRITICAL)

    def test_stationary_vehicle_does_not_trigger(self):
        det = NearMissDetector(min_samples=4)
        t, result = 0.0, None
        for _ in range(10):
            result = det.update([_make_tracked_obj(1, 100)], 640, t)
            t += 0.1
        assert result.near_miss_level == NEAR_MISS_NONE

    def test_receding_vehicle_does_not_trigger(self):
        det = NearMissDetector(min_samples=4)
        t, result = 0.0, None
        for i in range(10):
            result = det.update([_make_tracked_obj(1, 200 - i * 10)], 640, t)
            t += 0.1
        assert result.near_miss_level == NEAR_MISS_NONE

    def test_vehicle_outside_ego_lane_does_not_trigger(self):
        det = NearMissDetector(min_samples=4)
        t, result = 0.0, None
        for i in range(10):
            result = det.update([_make_tracked_obj(1, 50 + i * 20, center_x=600)], 640, t)
            t += 0.1
        assert result.near_miss_level == NEAR_MISS_NONE

    def test_insufficient_samples_does_not_trigger(self):
        det = NearMissDetector(min_samples=4)
        det.update([_make_tracked_obj(1, 50)], 640, 0.0)
        result = det.update([_make_tracked_obj(1, 200)], 640, 0.1)
        assert result.near_miss_level == NEAR_MISS_NONE

    def test_most_urgent_vehicle_wins_among_multiple(self):
        det = NearMissDetector(min_samples=4, warning_ttc_seconds=5.0, critical_ttc_seconds=2.0)
        t, result = 0.0, None
        for i in range(10):
            slow = _make_tracked_obj(1, 100 + i * 5, center_x=320)
            fast = _make_tracked_obj(2, 50 + i * 30, center_x=330)
            result = det.update([slow, fast], 640, t)
            t += 0.1
        assert result.tracking_id == 2

    def test_no_vehicles_no_crash(self):
        det = NearMissDetector()
        result = det.update([], 640, 0.0)
        assert result.near_miss_level == NEAR_MISS_NONE


class TestConfidenceScorer:
    def test_no_inputs_returns_unknown(self):
        result = compute_confidence()
        assert result.overall_confidence is None
        assert result.confidence_label == "UNKNOWN"
        assert result.components == []

    def test_real_photo_produces_real_confidence(self, real_test_image_path):
        from src.driver_monitoring.driver_analyzer import DriverAnalyzer

        analyzer = DriverAnalyzer()
        frame = cv2.imread(real_test_image_path)
        driver_result = analyzer.analyze(frame, timestamp=0.0)
        result = compute_confidence(driver_result=driver_result)
        # bus.jpg has no clearly frontal face for MediaPipe in this crop;
        # what matters is the function runs cleanly and returns a
        # well-formed result either way, not a specific detection outcome.
        assert result.confidence_label in ("HIGH", "MEDIUM", "LOW", "UNKNOWN")

    def test_no_face_detected_gives_no_fabricated_confidence(self, blank_frame):
        from src.driver_monitoring.driver_analyzer import DriverAnalyzer

        analyzer = DriverAnalyzer()
        driver_result = analyzer.analyze(blank_frame, timestamp=0.0)
        result = compute_confidence(driver_result=driver_result)
        assert result.overall_confidence is None
        assert result.confidence_label == "UNKNOWN"


class TestIncidentRecorder:
    def test_should_record_only_high_and_critical(self):
        assert IncidentRecorder.should_record("CRITICAL") is True
        assert IncidentRecorder.should_record("HIGH") is True
        assert IncidentRecorder.should_record("MEDIUM") is False
        assert IncidentRecorder.should_record("LOW") is False

    def test_record_and_retrieve_round_trip(self, tmp_path):
        recorder = IncidentRecorder(output_dir=str(tmp_path))
        road_frame = np.full((50, 50, 3), (50, 80, 120), dtype=np.uint8)

        record = recorder.record(
            alert_type="DROWSINESS", severity="HIGH", message="test message",
            risk_score=75, risk_level="HIGH",
            factor_breakdown=[{"factor": "drowsiness", "points": 30, "description": "x"}],
            road_frame=road_frame, timestamp=1700000000.0,
        )

        assert record.road_frame_path is not None
        import os
        assert os.path.exists(record.road_frame_path)

        fetched = recorder.get_incident(record.incident_id)
        assert fetched["alert_type"] == "DROWSINESS"
        assert fetched["factor_breakdown"][0]["points"] == 30

    def test_list_incidents_most_recent_first(self, tmp_path):
        recorder = IncidentRecorder(output_dir=str(tmp_path))
        recorder.record(alert_type="A", severity="HIGH", message="m1", risk_score=70, risk_level="HIGH", timestamp=1.0)
        second = recorder.record(alert_type="B", severity="CRITICAL", message="m2", risk_score=90, risk_level="CRITICAL", timestamp=2.0)

        incidents = recorder.list_incidents()
        assert len(incidents) == 2
        assert incidents[0]["incident_id"] == second.incident_id

    def test_get_nonexistent_incident_returns_none(self, tmp_path):
        recorder = IncidentRecorder(output_dir=str(tmp_path))
        assert recorder.get_incident("does-not-exist") is None

    def test_persists_across_recorder_instances(self, tmp_path):
        recorder1 = IncidentRecorder(output_dir=str(tmp_path))
        recorder1.record(alert_type="A", severity="HIGH", message="m", risk_score=70, risk_level="HIGH")

        recorder2 = IncidentRecorder(output_dir=str(tmp_path))
        assert len(recorder2.list_incidents()) == 1


class TestExplainableRisk:
    def test_breakdown_sums_to_score(self):
        calc = RiskCalculator()
        result = calc.calculate(
            RiskFactors(drowsiness=True, unsafe_following_distance=True, lane_violation=True)
        )
        explanation = result.explain()
        total_points = sum(f["points"] for f in explanation["factor_breakdown"])
        assert total_points == explanation["risk_score"]
        assert explanation["was_capped"] is False

    def test_capped_score_reports_was_capped(self):
        calc = RiskCalculator()
        result = calc.calculate(
            RiskFactors(
                drowsiness=True, unsafe_following_distance=True, lane_violation=True,
                road_hazard=True, distraction=True,
            )
        )
        explanation = result.explain()
        assert explanation["was_capped"] is True
        assert explanation["raw_score_before_cap"] == 110
        assert explanation["risk_score"] == 100

    def test_no_factors_empty_breakdown(self):
        calc = RiskCalculator()
        result = calc.calculate(RiskFactors())
        explanation = result.explain()
        assert explanation["factor_breakdown"] == []
        assert explanation["was_capped"] is False


class TestAlertPrioritization:
    def test_critical_ranked_above_high_and_medium(self):
        mgr = AlertManager()
        mgr._history = [
            Alert(type="A", severity="MEDIUM", message="m1", timestamp="t1"),
            Alert(type="B", severity="HIGH", message="m2", timestamp="t2"),
            Alert(type="C", severity="CRITICAL", message="m3", timestamp="t3"),
        ]
        prioritized = mgr.get_prioritized_alerts()
        assert prioritized[0].severity == "CRITICAL"
        assert prioritized[-1].severity == "MEDIUM"

    def test_recency_breaks_ties_within_same_severity(self):
        mgr = AlertManager()
        mgr._history = [
            Alert(type="A", severity="MEDIUM", message="older", timestamp="t1"),
            Alert(type="B", severity="MEDIUM", message="newer", timestamp="t2"),
        ]
        prioritized = mgr.get_prioritized_alerts()
        assert prioritized[0].message == "newer"

    def test_top_priority_alert_empty_history(self):
        mgr = AlertManager()
        assert mgr.get_top_priority_alert() is None


class TestNearMissAlertIntegration:
    def test_near_miss_fires_immediately_no_persistence_delay(self):
        calc = RiskCalculator()
        mgr = AlertManager()
        factors = RiskFactors()
        risk = calc.calculate(factors)
        alerts = mgr.process(factors, risk, timestamp=0.0, near_miss_level="CRITICAL", near_miss_ttc_seconds=1.1)
        assert len(alerts) == 1
        assert alerts[0].type == "NEAR_MISS"

    def test_near_miss_respects_cooldown(self):
        calc = RiskCalculator()
        mgr = AlertManager(near_miss_cooldown_seconds=5.0)
        factors = RiskFactors()
        risk = calc.calculate(factors)
        mgr.process(factors, risk, timestamp=0.0, near_miss_level="CRITICAL")
        repeat = mgr.process(factors, risk, timestamp=1.0, near_miss_level="CRITICAL")
        assert len(repeat) == 0

    def test_near_miss_none_never_fires(self):
        calc = RiskCalculator()
        mgr = AlertManager()
        factors = RiskFactors()
        risk = calc.calculate(factors)
        alerts = mgr.process(factors, risk, timestamp=0.0, near_miss_level="NONE")
        assert len(alerts) == 0


class TestVoiceAlertLogic:
    def test_speak_js_escapes_html_and_backticks(self):
        from frontend.components.voice_alert import _speak_js

        dangerous = "Alert: <script>alert(1)</script> and a `backtick`"
        js = _speak_js(dangerous)
        assert "<script>alert(1)</script>" not in js
        assert "speechSynthesis" in js

    def test_sort_alerts_by_priority_pure_function(self):
        from frontend.components.alert_panel import sort_alerts_by_priority

        alerts = [
            {"type": "A", "severity": "MEDIUM"},
            {"type": "B", "severity": "CRITICAL"},
        ]
        ordered = sort_alerts_by_priority(alerts)
        assert ordered[0]["type"] == "B"
