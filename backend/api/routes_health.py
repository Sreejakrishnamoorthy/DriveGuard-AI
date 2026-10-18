"""
Health check endpoint.
"""

from fastapi import APIRouter, Request
from datetime import datetime, timezone

from src.utils.config import Config

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check(request: Request) -> dict:
    """
    Basic liveness/readiness check.

    Reports whether the process is up AND whether key subsystems
    (models, database) are actually available, so a 200 response doesn't
    silently hide a degraded backend.
    """
    service = request.app.state.analysis_service

    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": Config.validate_device(),
        "models": {
            "vehicle_detector": service.road_analyzer.vehicle_detector.model is not None,
            "vehicle_tracker": service.road_analyzer.vehicle_tracker.is_available(),
            "hazard_detector_mode": service.road_analyzer.hazard_detector.mode,
            "face_detector": service.driver_analyzer.face_detector.is_available(),
        },
        "database_connected": service.database_service.is_connected,
    }
