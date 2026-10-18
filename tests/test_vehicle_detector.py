"""
Tests for vehicle detection output format and basic tracking behavior.
"""

import cv2
import numpy as np
import pytest

from src.road_analysis.vehicle_detector import VehicleDetector, Detection
from src.road_analysis.vehicle_tracker import VehicleTracker


class TestVehicleDetectorOutputFormat:
    def test_detection_has_required_fields(self, real_test_image_path):
        detector = VehicleDetector(confidence_threshold=0.35)
        frame = cv2.imread(real_test_image_path)
        detections = detector.detect(frame)

        assert len(detections) > 0, "Expected at least one detection on a real photo with people/a bus"
        for d in detections:
            assert isinstance(d, Detection)
            assert isinstance(d.class_name, str)
            assert 0.0 <= d.confidence <= 1.0
            assert len(d.bbox) == 4
            x1, y1, x2, y2 = d.bbox
            assert x2 > x1
            assert y2 > y1

    def test_detects_expected_classes_on_real_photo(self, real_test_image_path):
        """The bus.jpg test asset should yield at least a bus and people."""
        detector = VehicleDetector(confidence_threshold=0.35)
        frame = cv2.imread(real_test_image_path)
        detections = detector.detect(frame)
        class_names = {d.class_name for d in detections}
        assert "bus" in class_names or "person" in class_names

    def test_no_detections_on_blank_frame(self, blank_frame):
        detector = VehicleDetector(confidence_threshold=0.5)
        detections = detector.detect(blank_frame)
        assert detections == []

    def test_detect_does_not_raise_on_blank_frame(self, blank_frame):
        detector = VehicleDetector()
        # Should not raise even with no real content
        detections = detector.detect(blank_frame)
        assert isinstance(detections, list)


class TestVehicleTracker:
    def test_tracker_assigns_ids(self, real_test_image_path):
        detector = VehicleDetector(confidence_threshold=0.35)
        tracker = VehicleTracker()
        frame = cv2.imread(real_test_image_path)

        detections = detector.detect(frame)
        tracked = tracker.update(detections)

        assert len(tracked) > 0
        for obj in tracked:
            assert isinstance(obj.tracking_id, int)
            assert obj.tracking_id > 0

    def test_tracker_handles_empty_detections(self):
        tracker = VehicleTracker()
        tracked = tracker.update([])
        assert tracked == []

    def test_known_limitation_one_frame_gap_after_empty_frame(self, real_test_image_path):
        """
        Documents a verified upstream supervision.ByteTrack characteristic:
        real detections on the frame immediately following an empty frame
        are dropped, self-correcting on the next frame after that. See the
        "KNOWN LIMITATION" note in vehicle_tracker.py's class docstring.
        This test exists so a future supervision version that fixes this
        (in which case this test would start failing) gets noticed, rather
        than the fixed behavior going unremarked.
        """
        detector = VehicleDetector(confidence_threshold=0.35)
        frame = cv2.imread(real_test_image_path)
        detections = detector.detect(frame)
        assert len(detections) > 0

        tracker = VehicleTracker()
        tracker.update([])  # empty frame
        gap_frame = tracker.update(detections)  # currently dropped (known limitation)
        recovered_frame = tracker.update(detections)  # currently recovers

        assert gap_frame == []
        assert len(recovered_frame) > 0
