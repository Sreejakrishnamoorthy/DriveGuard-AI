"""
Analysis endpoints: frame/video analysis and current driver/road status.
"""

import os
import tempfile
import time
import uuid
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status

from backend.schemas.analysis import (
    ConfidenceSchema,
    DriverStatusSchema,
    FrameAnalysisResponse,
    RiskExplanationSchema,
    RoadStatusSchema,
    VideoAnalysisSummarySchema,
)
from src.utils.config import Config
from src.utils.logger import logger

router = APIRouter(prefix="/api", tags=["analysis"])


def _decode_upload_to_frame(contents: bytes) -> np.ndarray:
    """Decode uploaded image bytes into a BGR OpenCV frame."""
    arr = np.frombuffer(contents, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not decode uploaded file as an image.",
        )
    return frame


@router.post("/analyze/frame", response_model=FrameAnalysisResponse)
async def analyze_frame(
    request: Request,
    road_frame: Optional[UploadFile] = File(default=None),
    driver_frame: Optional[UploadFile] = File(default=None),
) -> FrameAnalysisResponse:
    """
    Analyze a single road frame and/or driver frame.

    At least one of road_frame / driver_frame must be provided. Updates
    the service's current-state snapshot used by the status/risk endpoints.
    """
    if road_frame is None and driver_frame is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide at least one of road_frame or driver_frame.",
        )

    service = request.app.state.analysis_service

    road_np = _decode_upload_to_frame(await road_frame.read()) if road_frame else None
    driver_np = _decode_upload_to_frame(await driver_frame.read()) if driver_frame else None

    result = service.analyze_frame(road_frame=road_np, driver_frame=driver_np, timestamp=time.time())

    return FrameAnalysisResponse(
        road=RoadStatusSchema(**result["road"]) if result["road"] else None,
        driver=DriverStatusSchema(**result["driver"]) if result["driver"] else None,
        phone_detected=result["phone_detected"],
        risk_score=result["risk_score"],
        risk_level=result["risk_level"],
        risk_explanation=RiskExplanationSchema(**result["risk_explanation"]),
        confidence=ConfidenceSchema(**result["confidence"]),
        explanation=result["explanation"],
    )


@router.post("/analyze/video", response_model=VideoAnalysisSummarySchema)
async def analyze_video(
    request: Request,
    road_video: Optional[UploadFile] = File(default=None),
    driver_video: Optional[UploadFile] = File(default=None),
    frame_skip: int = 1,
) -> VideoAnalysisSummarySchema:
    """
    Analyze an uploaded video (road and/or driver), processing it
    synchronously frame-by-frame and returning an aggregate summary.

    This runs synchronously within the request — for long videos in a
    production deployment, this endpoint would typically hand off to a
    background task queue instead; documented here as a known scope
    boundary for this project, not silently pretended away.
    """
    if road_video is None and driver_video is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide at least one of road_video or driver_video.",
        )

    service = request.app.state.analysis_service
    service.reset()

    session_id = service.database_service.create_session(
        {"driver_status": "PROCESSING", "road_status": "PROCESSING"}
    )
    service.state.active_session_id = session_id

    tmp_paths = []
    road_cap = None
    driver_cap = None

    try:
        if road_video is not None:
            road_path = os.path.join(tempfile.gettempdir(), f"road_{uuid.uuid4().hex}.mp4")
            with open(road_path, "wb") as f:
                f.write(await road_video.read())
            tmp_paths.append(road_path)
            road_cap = cv2.VideoCapture(road_path)
            if not road_cap.isOpened():
                raise HTTPException(status_code=400, detail="Could not open road_video.")

        if driver_video is not None:
            driver_path = os.path.join(tempfile.gettempdir(), f"driver_{uuid.uuid4().hex}.mp4")
            with open(driver_path, "wb") as f:
                f.write(await driver_video.read())
            tmp_paths.append(driver_path)
            driver_cap = cv2.VideoCapture(driver_path)
            if not driver_cap.isOpened():
                raise HTTPException(status_code=400, detail="Could not open driver_video.")

        fps = 30.0
        if road_cap is not None:
            fps = road_cap.get(cv2.CAP_PROP_FPS) or 30.0
        elif driver_cap is not None:
            fps = driver_cap.get(cv2.CAP_PROP_FPS) or 30.0

        frame_idx = 0
        processed = 0
        risk_scores = []
        total_new_alerts = 0
        t = 0.0
        dt = 1.0 / fps

        while True:
            road_frame = None
            driver_frame = None

            if road_cap is not None:
                ok_r, road_frame = road_cap.read()
                if not ok_r:
                    road_frame = None
                    road_cap = None if driver_cap is None else road_cap

            if driver_cap is not None:
                ok_d, driver_frame = driver_cap.read()
                if not ok_d:
                    driver_frame = None
                    driver_cap = None if road_cap is None else driver_cap

            if road_frame is None and driver_frame is None:
                break

            frame_idx += 1
            if frame_skip > 1 and frame_idx % frame_skip != 0:
                t += dt
                continue

            result = service.analyze_frame(
                road_frame=road_frame,
                driver_frame=driver_frame,
                timestamp=t,
                generate_explanation=False,
            )
            risk_scores.append(result["risk_score"])
            total_new_alerts += len(result["new_alerts"])
            processed += 1
            t += dt

        avg_risk = sum(risk_scores) / len(risk_scores) if risk_scores else 0.0
        peak_risk = max(risk_scores) if risk_scores else 0
        from src.risk_engine.risk_rules import get_risk_level

        peak_level = get_risk_level(peak_risk)

        if session_id:
            service.database_service.end_session(session_id, peak_risk, peak_level)

        return VideoAnalysisSummarySchema(
            session_id=session_id or "no-session-db-unavailable",
            frames_processed=processed,
            duration_seconds=round(t, 2),
            average_risk_score=round(avg_risk, 2),
            peak_risk_score=peak_risk,
            peak_risk_level=peak_level,
            total_alerts=total_new_alerts,
            output_video_path=None,
        )

    finally:
        for p in tmp_paths:
            try:
                os.remove(p)
            except OSError:
                pass


@router.get("/driver/status", response_model=DriverStatusSchema)
async def driver_status(request: Request) -> DriverStatusSchema:
    """Get the most recent driver-monitoring status."""
    service = request.app.state.analysis_service
    return DriverStatusSchema(**service.get_driver_status())


@router.get("/road/status", response_model=RoadStatusSchema)
async def road_status(request: Request) -> RoadStatusSchema:
    """Get the most recent road-analysis status."""
    service = request.app.state.analysis_service
    return RoadStatusSchema(**service.get_road_status())
