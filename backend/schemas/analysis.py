"""
Pydantic schemas for road/driver analysis API endpoints.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class VehicleSchema(BaseModel):
    """A single tracked vehicle/pedestrian in a road analysis result."""

    tracking_id: int
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: List[float] = Field(min_length=4, max_length=4)
    center_x: float
    center_y: float


class LaneSchema(BaseModel):
    """Lane detection result."""

    lane_detected: bool
    lane_status: str
    lane_departure: bool


class HazardItemSchema(BaseModel):
    """A single detected road hazard."""

    type: str
    confidence: float = Field(ge=0.0, le=1.0)


class VehicleSummarySchema(BaseModel):
    """Counts of tracked objects by category."""

    total: int = 0
    cars: int = 0
    trucks: int = 0
    buses: int = 0
    motorcycles: int = 0
    bicycles: int = 0
    pedestrians: int = 0


class NearMissSchema(BaseModel):
    """Phase 23 — near-miss / time-to-collision assessment."""

    near_miss_level: str = "NONE"
    time_to_collision_seconds: Optional[float] = None


class RoadStatusSchema(BaseModel):
    """Full road-analysis output for one frame."""

    vehicles: List[VehicleSchema] = []
    lane: LaneSchema
    hazards: List[HazardItemSchema] = []
    hazard_detected: bool = False
    following_distance: str
    vehicle_summary: VehicleSummarySchema = VehicleSummarySchema()
    near_miss: NearMissSchema = NearMissSchema()


class DriverStatusSchema(BaseModel):
    """Full driver-analysis output for one frame."""

    face_detected: bool
    eye_state: str = "UNKNOWN"
    drowsiness: bool = False
    drowsiness_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    attention: str = "UNKNOWN"
    driver_status: str
    phone_detected: bool = False


class ConfidenceComponentSchema(BaseModel):
    """Phase 26 — one input into the overall confidence assessment."""

    component: str
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    note: str


class ConfidenceSchema(BaseModel):
    """Phase 26 — overall confidence assessment for one frame's analysis."""

    overall_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    confidence_label: str = "UNKNOWN"
    components: List[ConfidenceComponentSchema] = []


class RiskFactorBreakdownSchema(BaseModel):
    """Phase 28 — one factor's point contribution to the risk score."""

    factor: str
    points: int
    description: str


class RiskExplanationSchema(BaseModel):
    """Phase 28 — full explainable-risk breakdown."""

    risk_score: int = Field(ge=0, le=100)
    risk_level: str
    raw_score_before_cap: int
    was_capped: bool
    factor_breakdown: List[RiskFactorBreakdownSchema] = []


class FrameAnalysisResponse(BaseModel):
    """Response for POST /api/analyze/frame."""

    road: Optional[RoadStatusSchema] = None
    driver: Optional[DriverStatusSchema] = None
    phone_detected: bool = False
    risk_score: int = Field(ge=0, le=100)
    risk_level: str
    risk_explanation: Optional[RiskExplanationSchema] = None
    confidence: Optional[ConfidenceSchema] = None
    explanation: Optional[str] = None


class VideoAnalysisSummarySchema(BaseModel):
    """Response for POST /api/analyze/video — aggregate summary of a processed video."""

    session_id: str
    frames_processed: int
    duration_seconds: float
    average_risk_score: float
    peak_risk_score: int
    peak_risk_level: str
    total_alerts: int
    output_video_path: Optional[str] = None
