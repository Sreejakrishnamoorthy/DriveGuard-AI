"""
Incident Recorder — Phase 27.

When a high-severity alert fires, saves a reviewable snapshot: the
annotated road/driver frame(s) at that moment, plus the full risk
breakdown (Phase 28) and confidence assessment (Phase 26) that produced
it. This turns "something bad happened at some point during the video"
into a concrete, browsable record you can go back and look at.

Storage is plain files on disk (one folder per incident, containing a
JSON manifest and any available frame images) rather than a database
table, so it works even when MongoDB isn't connected — consistent with
this project's approach of not making optional infrastructure a hard
dependency for core functionality.
"""

import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from src.utils.config import Config
from src.utils.logger import logger

DEFAULT_INCIDENTS_DIR = os.path.join(Config.OUTPUTS_DIR, "incidents")

# Only these severities are worth permanently recording — LOW/MEDIUM
# alerts are common and not "incidents" in the reviewable sense.
RECORDABLE_SEVERITIES = {"HIGH", "CRITICAL"}


@dataclass
class IncidentRecord:
    """A single saved incident."""

    incident_id: str
    timestamp: str
    alert_type: str
    severity: str
    message: str
    risk_score: int
    risk_level: str
    factor_breakdown: List[dict]
    road_status: Optional[dict]
    driver_status: Optional[dict]
    confidence: Optional[dict]
    road_frame_path: Optional[str]
    driver_frame_path: Optional[str]

    def to_dict(self) -> dict:
        return {
            "incident_id": self.incident_id,
            "timestamp": self.timestamp,
            "alert_type": self.alert_type,
            "severity": self.severity,
            "message": self.message,
            "risk_score": self.risk_score,
            "risk_level": self.risk_level,
            "factor_breakdown": self.factor_breakdown,
            "road_status": self.road_status,
            "driver_status": self.driver_status,
            "confidence": self.confidence,
            "road_frame_path": self.road_frame_path,
            "driver_frame_path": self.driver_frame_path,
        }


class IncidentRecorder:
    """
    Saves and retrieves incident snapshots for high-severity alerts.
    """

    def __init__(self, output_dir: Optional[str] = None) -> None:
        self.output_dir = output_dir or DEFAULT_INCIDENTS_DIR
        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def should_record(severity: str) -> bool:
        """Whether an alert of this severity warrants a permanent incident record."""
        return severity in RECORDABLE_SEVERITIES

    def record(
        self,
        alert_type: str,
        severity: str,
        message: str,
        risk_score: int,
        risk_level: str,
        factor_breakdown: Optional[List[dict]] = None,
        road_status: Optional[dict] = None,
        driver_status: Optional[dict] = None,
        confidence: Optional[dict] = None,
        road_frame: Optional[np.ndarray] = None,
        driver_frame: Optional[np.ndarray] = None,
        timestamp: Optional[float] = None,
    ) -> IncidentRecord:
        """
        Save a new incident record.

        Args:
            alert_type, severity, message: from the Alert that triggered this
            risk_score, risk_level: current overall risk at the moment of recording
            factor_breakdown: RiskScoreResult.factor_breakdown (Phase 28), if available
            road_status, driver_status: plain dicts (to_dict() output) of current state
            confidence: ConfidenceResult.to_dict() output (Phase 26), if available
            road_frame, driver_frame: annotated frames to save as images, if available
            timestamp: epoch seconds; defaults to wall-clock time

        Returns:
            The saved IncidentRecord (also written to disk as manifest.json
            plus any provided frame images, under output_dir/<incident_id>/)
        """
        incident_id = str(uuid.uuid4())
        incident_dir = os.path.join(self.output_dir, incident_id)
        os.makedirs(incident_dir, exist_ok=True)

        iso_ts = datetime.fromtimestamp(
            timestamp if timestamp is not None else datetime.now(timezone.utc).timestamp(),
            tz=timezone.utc,
        ).isoformat()

        road_frame_path = None
        driver_frame_path = None

        try:
            if road_frame is not None:
                road_frame_path = os.path.join(incident_dir, "road_frame.jpg")
                cv2.imwrite(road_frame_path, road_frame)
            if driver_frame is not None:
                driver_frame_path = os.path.join(incident_dir, "driver_frame.jpg")
                cv2.imwrite(driver_frame_path, driver_frame)
        except Exception as e:
            logger.error(f"IncidentRecorder: failed to save frame image(s): {e}", exc_info=True)

        record = IncidentRecord(
            incident_id=incident_id,
            timestamp=iso_ts,
            alert_type=alert_type,
            severity=severity,
            message=message,
            risk_score=risk_score,
            risk_level=risk_level,
            factor_breakdown=factor_breakdown or [],
            road_status=road_status,
            driver_status=driver_status,
            confidence=confidence,
            road_frame_path=road_frame_path,
            driver_frame_path=driver_frame_path,
        )

        try:
            manifest_path = os.path.join(incident_dir, "manifest.json")
            with open(manifest_path, "w") as f:
                json.dump(record.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"IncidentRecorder: failed to write manifest: {e}", exc_info=True)

        logger.info(f"Incident recorded: {incident_id} ({severity} {alert_type})")
        return record

    def list_incidents(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        List recorded incidents, most recent first.

        Reads manifests from disk — works without any database. Returns
        [] if the incidents directory doesn't exist or is empty (never
        raises).
        """
        if not os.path.isdir(self.output_dir):
            return []

        try:
            entries = []
            for name in os.listdir(self.output_dir):
                manifest_path = os.path.join(self.output_dir, name, "manifest.json")
                if os.path.isfile(manifest_path):
                    try:
                        with open(manifest_path) as f:
                            entries.append(json.load(f))
                    except Exception as e:
                        logger.warning(f"IncidentRecorder: skipping unreadable manifest {manifest_path}: {e}")

            entries.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
            return entries[:limit]

        except Exception as e:
            logger.error(f"IncidentRecorder: failed to list incidents: {e}", exc_info=True)
            return []

    def get_incident(self, incident_id: str) -> Optional[Dict[str, Any]]:
        """Get a single incident's full manifest by ID, or None if not found."""
        manifest_path = os.path.join(self.output_dir, incident_id, "manifest.json")
        if not os.path.isfile(manifest_path):
            return None
        try:
            with open(manifest_path) as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"IncidentRecorder: failed to read incident {incident_id}: {e}", exc_info=True)
            return None
