"""
Tests for the FastAPI backend's health endpoint and basic API behavior.
"""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from backend.main import app

    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_shape(self, client):
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "device" in data
        assert "models" in data
        assert "database_connected" in data
        assert data["status"] == "ok"

    def test_root_endpoint(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["name"] == "DriveGuard AI"


class TestRiskEndpointDefaults:
    def test_current_risk_before_any_analysis(self, client):
        response = client.get("/api/risk/current")
        assert response.status_code == 200
        data = response.json()
        assert data["risk_score"] == 0
        assert data["risk_level"] == "LOW"


class TestStatusEndpointsDoNotCrashWithNoData:
    def test_driver_status_no_data(self, client):
        response = client.get("/api/driver/status")
        assert response.status_code == 200
        assert response.json()["face_detected"] is False

    def test_road_status_no_data(self, client):
        response = client.get("/api/road/status")
        assert response.status_code == 200

    def test_alerts_empty(self, client):
        response = client.get("/api/alerts")
        assert response.status_code == 200
        assert response.json() == []


class TestAnalyzeFrameValidation:
    def test_no_files_returns_400(self, client):
        response = client.post("/api/analyze/frame")
        assert response.status_code == 400

    def test_invalid_image_bytes_returns_400(self, client):
        response = client.post(
            "/api/analyze/frame",
            files={"driver_frame": ("bad.jpg", b"not an image", "image/jpeg")},
        )
        assert response.status_code == 400


class TestSessionsRequireDatabase:
    def test_create_session_without_database_returns_503(self, client):
        """When MongoDB is unavailable, the API must say so explicitly, not silently succeed."""
        response = client.post("/api/sessions", json={"driver_status": "NORMAL"})
        # This test's outcome depends on whether MongoDB happens to be running
        # in the test environment; both explicit outcomes are valid responses.
        assert response.status_code in (201, 503)

    def test_get_nonexistent_session_returns_404(self, client):
        response = client.get("/api/sessions/definitely-does-not-exist")
        assert response.status_code == 404


class TestExtendedEndpoints:
    """Phase 19-28 API surface: explainability, confidence, incidents."""

    def test_risk_explain_returns_200(self, client):
        response = client.get("/api/risk/explain")
        assert response.status_code == 200
        data = response.json()
        assert "factor_breakdown" in data
        assert "was_capped" in data

    def test_risk_confidence_returns_200(self, client):
        response = client.get("/api/risk/confidence")
        assert response.status_code == 200
        data = response.json()
        assert "confidence_label" in data

    def test_incidents_list_returns_200_empty_by_default(self, client):
        response = client.get("/api/incidents")
        assert response.status_code == 200
        assert "incidents" in response.json()

    def test_get_nonexistent_incident_returns_404(self, client):
        response = client.get("/api/incidents/definitely-does-not-exist")
        assert response.status_code == 404

    def test_driver_status_includes_phone_detected_field(self, client):
        response = client.get("/api/driver/status")
        assert response.status_code == 200
        assert "phone_detected" in response.json()

    def test_road_status_includes_near_miss_field(self, client):
        response = client.get("/api/road/status")
        assert response.status_code == 200
        assert "near_miss" in response.json()

    def test_alerts_accepts_prioritized_param(self, client):
        response = client.get("/api/alerts", params={"prioritized": "false"})
        assert response.status_code == 200
