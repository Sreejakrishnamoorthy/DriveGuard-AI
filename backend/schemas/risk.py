"""
Pydantic schemas for risk-score and alert API endpoints.
"""

from typing import Optional
from pydantic import BaseModel, Field


class RiskScoreSchema(BaseModel):
    """Current risk score and level."""

    risk_score: int = Field(ge=0, le=100)
    risk_level: str
    explanation: Optional[str] = None


class AlertSchema(BaseModel):
    """A single structured alert."""

    type: str
    severity: str
    message: str
    timestamp: str
