"""
Tests for driver-monitoring temporal logic (drowsiness) and pure
classification logic (attention), independent of camera/photo quality.
"""

from src.driver_monitoring.drowsiness_detector import DrowsinessDetector
from src.driver_monitoring.attention_detector import (
    ATTENTION_ATTENTIVE,
    ATTENTION_DISTRACTED,
)
from src.driver_monitoring.eye_detector import compute_ear


class TestEyeAspectRatio:
    def test_open_eye_shape_has_high_ear(self):
        open_eye = [(0, 0), (10, -15), (20, -15), (30, 0), (20, 15), (10, 15)]
        assert compute_ear(open_eye) > 0.5

    def test_closed_eye_shape_has_low_ear(self):
        closed_eye = [(0, 0), (10, -2), (20, -2), (30, 0), (20, 2), (10, 2)]
        assert compute_ear(closed_eye) < 0.2


class TestDrowsinessDetector:
    def test_single_frame_blink_does_not_trigger(self):
        det = DrowsinessDetector(closed_duration_threshold_seconds=1.5, perclos_window_seconds=10, perclos_threshold=0.4)
        t = 0.0
        det.update("OPEN", t); t += 0.1
        det.update("CLOSED", t); t += 0.1
        result = det.update("OPEN", t)
        assert result.drowsiness is False

    def test_sustained_closure_under_threshold_does_not_trigger(self):
        det = DrowsinessDetector(closed_duration_threshold_seconds=1.5, perclos_window_seconds=10, perclos_threshold=0.4)
        t = 0.0
        result = None
        for _ in range(10):  # 1.0s closed, under the 1.5s threshold
            result = det.update("CLOSED", t)
            t += 0.1
        assert result.drowsiness is False

    def test_sustained_closure_past_threshold_triggers(self):
        det = DrowsinessDetector(closed_duration_threshold_seconds=1.5, perclos_window_seconds=10, perclos_threshold=0.4)
        t = 0.0
        result = None
        for _ in range(17):  # 1.7s closed, past the 1.5s threshold
            result = det.update("CLOSED", t)
            t += 0.1
        assert result.drowsiness is True

    def test_reopening_eyes_clears_drowsiness(self):
        det = DrowsinessDetector(closed_duration_threshold_seconds=1.5, perclos_window_seconds=10, perclos_threshold=0.4)
        t = 0.0
        for _ in range(17):
            det.update("CLOSED", t)
            t += 0.1
        result = det.update("OPEN", t)
        assert result.drowsiness is False

    def test_perclos_does_not_trigger_on_insufficient_history(self):
        """Regression test: PERCLOS must not fire off partial history (e.g. 3s of a 10s window)."""
        det = DrowsinessDetector(closed_duration_threshold_seconds=1.5, perclos_window_seconds=10, perclos_threshold=0.4)
        t = 0.0
        result = None
        for _ in range(5):  # 5 * 0.6s = 3s, well under the 10s window
            for _ in range(4):
                result = det.update("CLOSED", t); t += 0.1
            for _ in range(2):
                result = det.update("OPEN", t); t += 0.1
        assert result.drowsiness is False

    def test_perclos_triggers_with_full_window_of_heavy_blinking(self):
        det = DrowsinessDetector(closed_duration_threshold_seconds=1.5, perclos_window_seconds=10, perclos_threshold=0.4)
        t = 0.0
        result = None
        for _ in range(20):  # 20 * 0.6s = 12s, exceeds the 10s window
            for _ in range(4):
                result = det.update("CLOSED", t); t += 0.1
            for _ in range(2):
                result = det.update("OPEN", t); t += 0.1
        assert result.drowsiness is True

    def test_unknown_state_does_not_break_consecutive_tracking(self):
        det = DrowsinessDetector()
        t = 0.0
        det.update("CLOSED", t); t += 0.1
        det.update("UNKNOWN", t); t += 0.1
        result = det.update("CLOSED", t)
        assert abs(result.consecutive_closed_seconds - 0.2) < 0.01


class TestAttentionClassification:
    """Pure threshold classification, independent of camera/pose-estimation quality."""

    @staticmethod
    def classify(yaw, pitch, yaw_threshold=30.0, pitch_threshold=25.0):
        is_attentive = abs(yaw) <= yaw_threshold and abs(pitch) <= pitch_threshold
        return ATTENTION_ATTENTIVE if is_attentive else ATTENTION_DISTRACTED

    def test_facing_forward_is_attentive(self):
        assert self.classify(0, 0) == ATTENTION_ATTENTIVE

    def test_small_natural_movement_is_attentive(self):
        assert self.classify(10, 5) == ATTENTION_ATTENTIVE

    def test_large_yaw_is_distracted(self):
        assert self.classify(45, 0) == ATTENTION_DISTRACTED

    def test_large_pitch_is_distracted(self):
        assert self.classify(0, 40) == ATTENTION_DISTRACTED
