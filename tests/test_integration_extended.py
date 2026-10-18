"""
Integration tests for AnalysisService with all extended features wired
together (Phases 19-28), and the API endpoints that expose them.
"""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from backend.main import app

    with TestClient(app) as c:
        yield c


class TestAnalysisServiceIntegration:
    def test_analyze_frame_includes_all_new_fields(self, real_test_image_path):
        import cv2
        from backend.services.analysis_service import AnalysisService

        svc = AnalysisService()
        frame = cv2.imread(real_test_image_path)
        result = svc.analyze_frame(road_frame=frame, driver_frame=None, timestamp=0.0)

        assert "phone_detected" in result
        assert "risk_explanation" in result
        assert "confidence" in result
        assert "factor_breakdown" in result["risk_explanation"]
        assert "near_miss" in result["road"]

    def test_get_confidence_before_any_analysis(self):
        from backend.services.analysis_service import AnalysisService

        svc = AnalysisService()
        confidence = svc.get_confidence()
        assert confidence["overall_confidence"] is None
        assert confidence["confidence_label"] == "UNKNOWN"

    def test_get_risk_explanation_before_any_analysis(self):
        from backend.services.analysis_service import AnalysisService

        svc = AnalysisService()
        explanation = svc.get_risk_explanation()
        assert explanation["risk_score"] == 0
        assert explanation["factor_breakdown"] == []

    def test_incidents_empty_by_default(self):
        from backend.services.analysis_service import AnalysisService

        svc = AnalysisService()
        assert svc.get_incidents() == []

    def test_alerts_prioritized_by_default(self):
        from backend.services.analysis_service import AnalysisService
        from src.risk_engine.alert_manager import Alert

        svc = AnalysisService()
        svc.alert_manager._history = [
            Alert(type="A", severity="MEDIUM", message="m1", timestamp="t1"),
            Alert(type="B", severity="CRITICAL", message="m2", timestamp="t2"),
        ]
        alerts = svc.get_alerts()
        assert alerts[0]["severity"] == "CRITICAL"


class TestNewAPIEndpoints:
    def test_risk_explain_endpoint(self, client):
        response = client.get("/api/risk/explain")
        assert response.status_code == 200
        data = response.json()
        assert "factor_breakdown" in data
        assert "was_capped" in data

    def test_risk_confidence_endpoint(self, client):
        response = client.get("/api/risk/confidence")
        assert response.status_code == 200
        data = response.json()
        assert "confidence_label" in data

    def test_incidents_list_endpoint(self, client):
        response = client.get("/api/incidents")
        assert response.status_code == 200
        data = response.json()
        assert "incidents" in data
        assert "count" in data

    def test_incidents_nonexistent_returns_404(self, client):
        response = client.get("/api/incidents/does-not-exist")
        assert response.status_code == 404

    def test_alerts_endpoint_accepts_prioritized_param(self, client):
        response = client.get("/api/alerts", params={"prioritized": False})
        assert response.status_code == 200

    def test_driver_status_includes_phone_detected(self, client):
        response = client.get("/api/driver/status")
        assert response.status_code == 200
        assert "phone_detected" in response.json()

    def test_road_status_includes_near_miss(self, client):
        response = client.get("/api/road/status")
        assert response.status_code == 200
        assert "near_miss" in response.json()

    def test_analyze_frame_response_includes_new_fields(self, client, real_test_image_path):
        with open(real_test_image_path, "rb") as f:
            response = client.post(
                "/api/analyze/frame", files={"road_frame": ("bus.jpg", f, "image/jpeg")}
            )
        assert response.status_code == 200
        data = response.json()
        assert "phone_detected" in data
        assert "risk_explanation" in data
        assert "confidence" in data
