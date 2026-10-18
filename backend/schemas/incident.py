"""
Pydantic schemas for incident-reconstruction API endpoints — Phase 27.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class IncidentSummarySchema(BaseModel):
    """One incident record as returned by the API."""

    incident_id: str
    timestamp: str
    alert_type: str
    severity: str
    message: str
    risk_score: int
    risk_level: str
    factor_breakdown: List[Dict[str, Any]] = []
    road_status: Optional[Dict[str, Any]] = None
    driver_status: Optional[Dict[str, Any]] = None
    confidence: Optional[Dict[str, Any]] = None
    road_frame_path: Optional[str] = None
    driver_frame_path: Optional[str] = None


class IncidentListResponse(BaseModel):
    """Response for GET /api/incidents."""

    incidents: List[IncidentSummarySchema]
    count: int
