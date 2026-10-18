"""
Session management endpoints.
"""

from fastapi import APIRouter, HTTPException, Request, status

from backend.schemas.session import SessionCreateRequest, SessionListResponse, SessionResponse

router = APIRouter(prefix="/api", tags=["sessions"])


@router.post("/sessions", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(request: Request, body: SessionCreateRequest) -> SessionResponse:
    """
    Create a new session record.

    If MongoDB is unavailable, returns 503 rather than silently pretending
    a session was created — callers need to know persistence isn't active.
    """
    service = request.app.state.analysis_service

    if not service.database_service.is_connected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database unavailable — cannot create a persistent session right now.",
        )

    session_id = service.database_service.create_session(
        {
            "driver_status": body.driver_status,
            "road_status": body.road_status,
            **({"metadata": body.metadata} if body.metadata else {}),
        }
    )
    service.state.active_session_id = session_id

    session = service.database_service.get_session(session_id)
    return SessionResponse(**session)


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(request: Request, limit: int = 50) -> SessionListResponse:
    """List recent sessions, most recent first."""
    service = request.app.state.analysis_service
    sessions = service.database_service.list_sessions(limit=limit)
    return SessionListResponse(
        sessions=[SessionResponse(**s) for s in sessions], count=len(sessions)
    )


@router.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(request: Request, session_id: str) -> SessionResponse:
    """Get a single session by ID."""
    service = request.app.state.analysis_service
    session = service.database_service.get_session(session_id)

    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found.",
        )

    return SessionResponse(**session)
