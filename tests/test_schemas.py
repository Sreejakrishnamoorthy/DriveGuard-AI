"""
Tests for Pydantic API schemas: validation, defaults, and rejection of
invalid data.
"""

import pytest
from pydantic import ValidationError

from backend.schemas.analysis import (
    DriverStatusSchema,
    LaneSchema,
    RoadStatusSchema,
    VehicleSchema,
    VehicleSummarySchema,
)
from backend.schemas.risk import AlertSchema, RiskScoreSchema
from backend.schemas.session import SessionCreateRequest, SessionResponse


class TestVehicleSchema:
    def test_valid_vehicle(self):
        v = VehicleSchema(
            tracking_id=1, class_name="car", confidence=0.9,
            bbox=[10.0, 20.0, 100.0, 200.0], center_x=55.0, center_y=110.0,
        )
        assert v.tracking_id == 1

    def test_confidence_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            VehicleSchema(
                tracking_id=1, class_name="car", confidence=1.5,
                bbox=[0, 0, 1, 1], center_x=0, center_y=0,
            )

    def test_bbox_wrong_length_rejected(self):
        with pytest.raises(ValidationError):
            VehicleSchema(
                tracking_id=1, class_name="car", confidence=0.5,
                bbox=[0, 0, 1], center_x=0, center_y=0,
            )


class TestRiskScoreSchema:
    def test_valid_score(self):
        r = RiskScoreSchema(risk_score=75, risk_level="HIGH")
        assert r.risk_score == 75

    def test_score_above_100_rejected(self):
        with pytest.raises(ValidationError):
            RiskScoreSchema(risk_score=150, risk_level="CRITICAL")

    def test_negative_score_rejected(self):
        with pytest.raises(ValidationError):
            RiskScoreSchema(risk_score=-5, risk_level="LOW")


class TestAlertSchema:
    def test_valid_alert(self):
        a = AlertSchema(type="DROWSINESS", severity="HIGH", message="test", timestamp="2026-01-01T00:00:00")
        assert a.type == "DROWSINESS"

    def test_missing_field_rejected(self):
        with pytest.raises(ValidationError):
            AlertSchema(type="DROWSINESS", severity="HIGH")


class TestRoadStatusSchema:
    def test_valid_full_road_status(self):
        r = RoadStatusSchema(
            vehicles=[],
            lane=LaneSchema(lane_detected=True, lane_status="CENTERED", lane_departure=False),
            hazards=[],
            hazard_detected=False,
            following_distance="SAFE",
            vehicle_summary=VehicleSummarySchema(),
        )
        assert r.following_distance == "SAFE"

    def test_defaults_apply(self):
        r = RoadStatusSchema(
            lane=LaneSchema(lane_detected=False, lane_status="UNKNOWN", lane_departure=False),
            following_distance="SAFE",
        )
        assert r.vehicles == []
        assert r.hazard_detected is False


class TestDriverStatusSchema:
    def test_valid_driver_status(self):
        d = DriverStatusSchema(face_detected=True, driver_status="NORMAL")
        assert d.driver_status == "NORMAL"
        assert d.drowsiness_probability == 0.0  # default

    def test_drowsiness_probability_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            DriverStatusSchema(face_detected=True, driver_status="NORMAL", drowsiness_probability=2.0)


class TestSessionSchemas:
    def test_create_request_defaults(self):
        req = SessionCreateRequest()
        assert req.driver_status == "UNKNOWN"

    def test_session_response_requires_session_id(self):
        with pytest.raises(ValidationError):
            SessionResponse(start_time="2026-01-01T00:00:00")
