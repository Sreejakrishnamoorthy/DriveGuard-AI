"""
Tests for lane detection and hazard detection on synthetic scenes with
known, controllable ground truth (a real photo can't easily prove a
lane detector's departure logic without hand-verifying every frame, so
these use geometrically-constructed test scenes instead — see
src/road_analysis/lane_detector.py and hazard_detector.py for the
matching real-photo verification notes from development).
"""

import cv2
import numpy as np
import pytest

from src.road_analysis.lane_detector import LaneDetector
from src.road_analysis.hazard_detector import HazardDetector


def _make_road_frame(width=640, height=480, center_offset=0):
    """Build a synthetic road scene with perspective lane lines for lane-detector tests."""
    frame = np.full((height, width, 3), (60, 60, 60), dtype=np.uint8)
    cv2.rectangle(frame, (0, 0), (width, int(height * 0.55)), (150, 120, 90), -1)
    horizon_y = int(height * 0.58)
    bottom_y = height

    lane_half_width_bottom = 220
    lane_half_width_top = 25
    center_x_bottom = width // 2 + center_offset
    center_x_top = width // 2

    cv2.line(
        frame,
        (center_x_bottom - lane_half_width_bottom, bottom_y),
        (center_x_top - lane_half_width_top, horizon_y),
        (255, 255, 255),
        8,
    )
    cv2.line(
        frame,
        (center_x_bottom + lane_half_width_bottom, bottom_y),
        (center_x_top + lane_half_width_top, horizon_y),
        (255, 255, 255),
        8,
    )
    return frame


class TestLaneDetector:
    def test_centered_lane_reports_centered(self):
        detector = LaneDetector()
        frame = _make_road_frame(center_offset=0)
        # Run a few times to let the smoothing window settle
        for _ in range(6):
            result = detector.detect(frame)
        assert result.lane_detected is True
        assert result.lane_status == "CENTERED"
        assert result.lane_departure is False

    def test_significant_drift_triggers_departure(self):
        detector = LaneDetector()
        # Offset chosen empirically to clear departure_threshold_ratio=0.65
        # with margin (0.35 was the original conservative default; it was
        # deliberately raised during real-dashcam-footage tuning to reduce
        # false positives — see lane_detector.py constructor docstring).
        frame = _make_road_frame(center_offset=180)
        for _ in range(6):
            result = detector.detect(frame)
        assert result.lane_departure is True

    def test_no_lines_reports_not_detected(self):
        detector = LaneDetector()
        blank = np.full((480, 640, 3), 80, dtype=np.uint8)
        result = detector.detect(blank)
        assert result.lane_detected is False
        assert result.lane_status == "UNKNOWN"

    def test_never_raises_on_blank_frame(self):
        detector = LaneDetector()
        blank = np.full((480, 640, 3), 80, dtype=np.uint8)
        result = detector.detect(blank)  # should not raise
        assert result is not None


class TestHazardDetector:
    def test_no_model_configured_uses_heuristic_fallback(self):
        detector = HazardDetector(model_path=None)
        assert detector.mode == "heuristic_fallback"
        assert detector.is_using_pretrained_model() is False

    def test_clean_road_no_false_positive(self):
        detector = HazardDetector()
        clean_road = np.full((480, 640, 3), (70, 70, 70), dtype=np.uint8)
        cv2.rectangle(clean_road, (0, 0), (640, 260), (150, 120, 90), -1)
        result = detector.detect(clean_road)
        assert result.hazard_detected is False

    def test_dark_anomaly_detected(self):
        detector = HazardDetector()
        # A road background around gray-value 100 (typical daylight asphalt)
        # with a genuinely dark shadow/pothole-like region. The darkness
        # threshold was deliberately raised during real-footage tuning
        # (see hazard_detector.py constructor docstring) to avoid false
        # positives on hood reflections; this scenario has enough contrast
        # to clear that threshold the way a real dark hazard would.
        road = np.full((480, 640, 3), (100, 100, 100), dtype=np.uint8)
        cv2.rectangle(road, (0, 0), (640, 260), (150, 120, 90), -1)
        cv2.ellipse(road, (320, 400), (60, 35), 0, 0, 360, (10, 10, 10), -1)
        result = detector.detect(road)
        assert result.hazard_detected is True
        assert len(result.hazards) > 0
        assert result.hazards[0].confidence <= 1.0

    def test_result_tags_heuristic_source(self):
        detector = HazardDetector()
        road = np.full((480, 640, 3), (100, 100, 100), dtype=np.uint8)
        cv2.ellipse(road, (320, 300), (60, 35), 0, 0, 360, (10, 10, 10), -1)
        result = detector.detect(road)
        assert len(result.hazards) > 0, "Expected at least one detected hazard to check its source tag"
        for hazard in result.hazards:
            assert hazard.source == "heuristic_fallback"

    def test_never_raises_on_blank_frame(self):
        detector = HazardDetector()
        blank = np.full((480, 640, 3), 80, dtype=np.uint8)
        result = detector.detect(blank)
        assert result is not None
