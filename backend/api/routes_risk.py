"""
Risk score, alerts, explainability, and confidence endpoints.
"""

from typing import List

from fastapi import APIRouter, Request

from backend.schemas.risk import AlertSchema, RiskScoreSchema
from backend.schemas.analysis import ConfidenceSchema, RiskExplanationSchema

router = APIRouter(prefix="/api", tags=["risk"])


@router.get("/risk/current", response_model=RiskScoreSchema)
async def current_risk(request: Request) -> RiskScoreSchema:
    """Get the current risk score, level, and latest explanation."""
    service = request.app.state.analysis_service
    return RiskScoreSchema(**service.get_current_risk())


@router.get("/risk/explain", response_model=RiskExplanationSchema)
async def explain_risk(request: Request) -> RiskExplanationSchema:
    """
    Phase 28 — Explainable Risk.

    Get a transparent breakdown of exactly which factors contributed how
    many points to the current risk score, instead of just the total.
    """
    service = request.app.state.analysis_service
    return RiskExplanationSchema(**service.get_risk_explanation())


@router.get("/risk/confidence", response_model=ConfidenceSchema)
async def risk_confidence(request: Request) -> ConfidenceSchema:
    """
    Phase 26 — Confidence Scores.

    Get how much confidence the current assessment should carry, broken
    down by the detection components that fed into it (vehicle detection,
    face detection, eye-state margin, hazard-detector mode, etc.).
    """
    service = request.app.state.analysis_service
    return ConfidenceSchema(**service.get_confidence())


@router.get("/alerts", response_model=List[AlertSchema])
async def get_alerts(request: Request, limit: int = 50, prioritized: bool = True) -> List[AlertSchema]:
    """
    Get recent alerts.

    Phase 20 — Alert Prioritization: by default (prioritized=true),
    ordered by severity (most urgent first) rather than plain recency.
    Pass prioritized=false for simple chronological order instead.
    """
    service = request.app.state.analysis_service
    return [AlertSchema(**a) for a in service.get_alerts(limit=limit, prioritized=prioritized)]
