"""
Incident-reconstruction endpoints — Phase 27.
"""

from fastapi import APIRouter, HTTPException, Request, status

from backend.schemas.incident import IncidentListResponse, IncidentSummarySchema

router = APIRouter(prefix="/api", tags=["incidents"])


@router.get("/incidents", response_model=IncidentListResponse)
async def list_incidents(request: Request, limit: int = 50) -> IncidentListResponse:
    """
    List recorded incidents, most recent first.

    An incident is automatically saved whenever a HIGH or CRITICAL
    severity alert fires (drowsiness, near-miss, unsafe distance, or
    overall high risk), capturing the annotated frame(s) and full risk
    breakdown at that moment for later review.
    """
    service = request.app.state.analysis_service
    incidents = service.get_incidents(limit=limit)
    return IncidentListResponse(
        incidents=[IncidentSummarySchema(**inc) for inc in incidents], count=len(incidents)
    )


@router.get("/incidents/{incident_id}", response_model=IncidentSummarySchema)
async def get_incident(request: Request, incident_id: str) -> IncidentSummarySchema:
    """Get a single incident's full record by ID."""
    service = request.app.state.analysis_service
    incident = service.get_incident(incident_id)

    if incident is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Incident '{incident_id}' not found.",
        )

    return IncidentSummarySchema(**incident)
