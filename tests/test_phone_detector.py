"""
Tests for phone detection (Phase 25) — reuses the shared YOLO model's
built-in "cell phone" COCO class.
"""

import cv2
import numpy as np

from src.driver_monitoring.phone_detector import PhoneDetector, CELL_PHONE_CLASS_NAME


class TestPhoneDetector:
    def test_model_has_cell_phone_class(self):
        detector = PhoneDetector()
        detector.load_model()
        assert detector.is_available() is True
        assert detector.class_names[detector._phone_class_id] == CELL_PHONE_CLASS_NAME

    def test_no_false_positive_on_real_photo_without_phone(self, real_test_image_path):
        detector = PhoneDetector()
        frame = cv2.imread(real_test_image_path)
        result = detector.detect(frame)
        assert result.phone_detected is False

    def test_no_crash_on_blank_frame(self, blank_frame):
        detector = PhoneDetector()
        result = detector.detect(blank_frame)
        assert result.phone_detected is False

    def test_result_shape(self, blank_frame):
        detector = PhoneDetector()
        result = detector.detect(blank_frame)
        d = result.to_dict()
        assert "phone_detected" in d
        assert "confidence" in d

    def test_draw_detection_no_crash_when_no_phone(self, blank_frame):
        detector = PhoneDetector()
        result = detector.detect(blank_frame)
        annotated = detector.draw_detection(blank_frame, result)
        assert annotated is not None
