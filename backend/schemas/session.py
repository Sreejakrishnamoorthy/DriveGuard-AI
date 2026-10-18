"""
Pydantic schemas for session management API endpoints.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class SessionCreateRequest(BaseModel):
    """Request body for POST /api/sessions."""

    driver_status: Optional[str] = "UNKNOWN"
    road_status: Optional[str] = "UNKNOWN"
    metadata: Optional[Dict[str, Any]] = None


class SessionResponse(BaseModel):
    """A session record as returned by the API."""

    session_id: str
    start_time: str
    end_time: Optional[str] = None
    risk_score: int = 0
    risk_level: str = "LOW"
    driver_status: str = "UNKNOWN"
    road_status: str = "UNKNOWN"
    alerts: List[Dict[str, Any]] = []
    timestamps: Dict[str, Any] = {}


class SessionListResponse(BaseModel):
    """Response for GET /api/sessions."""

    sessions: List[SessionResponse]
    count: int
