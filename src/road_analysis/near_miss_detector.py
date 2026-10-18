"""
Near-Miss Detector — Phase 23.

Estimates a rough Time-To-Collision (TTC) for the vehicle directly ahead
by tracking how fast its bounding-box height grows across frames (an
object getting closer occupies a growing fraction of the frame).

IMPORTANT — monocular limitation (same honesty standard as
road_analyzer.py's following-distance estimate): this is a RELATIVE,
uncalibrated approximation, not a physically precise collision-timing
system. Real TTC estimation in production ADAS systems uses radar/lidar
or a calibrated stereo/mono camera with known focal length and target
dimensions. Here, TTC is approximated purely from the rate of bounding-
box growth, which correlates with — but is not equal to — true physical
time-to-collision. Treat NEAR_MISS as a directional warning signal, not
a certified safety measurement.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from src.road_analysis.vehicle_tracker import TrackedObject
from src.utils.logger import logger

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle"}

NEAR_MISS_NONE = "NONE"
NEAR_MISS_WARNING = "WARNING"
NEAR_MISS_CRITICAL = "CRITICAL"


@dataclass
class NearMissResult:
    """Result of near-miss / time-to-collision analysis for one frame."""

    near_miss_level: str = NEAR_MISS_NONE
    time_to_collision_seconds: Optional[float] = None
    tracking_id: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "near_miss_level": self.near_miss_level,
            "time_to_collision_seconds": (
                round(self.time_to_collision_seconds, 2)
                if self.time_to_collision_seconds is not None
                else None
            ),
        }


class NearMissDetector:
    """
    Tracks bounding-box height history per tracked object and estimates
    time-to-collision for whichever ego-lane vehicle is closing fastest.
    """

    def __init__(
        self,
        history_window_seconds: float = 2.0,
        min_samples: int = 4,
        warning_ttc_seconds: float = 2.5,
        critical_ttc_seconds: float = 1.2,
        ego_lane_ratio: float = 0.5,
        min_growth_rate: float = 0.5,
    ) -> None:
        """
        Initialize the near-miss detector.

        Args:
            history_window_seconds: How much recent history (seconds) to
                keep per tracked object for estimating growth rate.
            min_samples: Minimum number of samples required before
                attempting a TTC estimate (avoids noisy single-pair estimates).
            warning_ttc_seconds: TTC below this triggers a WARNING level.
            critical_ttc_seconds: TTC below this triggers a CRITICAL level.
            ego_lane_ratio: Fraction of frame width (centered) considered
                "directly ahead," matching RoadAnalyzer's following-distance logic.
            min_growth_rate: Minimum bbox-height growth rate (px/sec) to
                consider an object "approaching" at all — filters out
                slow drift/noise from being reported as a near miss.
        """
        self.history_window = history_window_seconds
        self.min_samples = min_samples
        self.warning_ttc = warning_ttc_seconds
        self.critical_ttc = critical_ttc_seconds
        self.ego_lane_ratio = ego_lane_ratio
        self.min_growth_rate = min_growth_rate

        self._history: Dict[int, Deque[Tuple[float, float]]] = {}

    def _estimate_growth_rate(self, samples: Deque[Tuple[float, float]]) -> Optional[float]:
        """
        Estimate bbox-height growth rate (px/sec) via a simple linear fit
        over recent (timestamp, height) samples.
        """
        if len(samples) < self.min_samples:
            return None

        times = np.array([t for t, _ in samples])
        heights = np.array([h for _, h in samples])

        if times.max() == times.min():
            return None

        # Simple linear regression slope: height = slope * time + intercept
        slope, _ = np.polyfit(times, heights, 1)
        return float(slope)

    def update(
        self, tracked_objects: List[TrackedObject], frame_width: int, timestamp: float
    ) -> NearMissResult:
        """
        Update per-object history and return the most urgent near-miss
        assessment for this frame (if any).

        Args:
            tracked_objects: Currently tracked objects (from VehicleTracker)
            frame_width: Width of the frame in pixels (for ego-lane filtering)
            timestamp: Current timestamp in seconds

        Returns:
            NearMissResult for the most time-critical ego-lane vehicle,
            or NEAR_MISS_NONE if nothing qualifies.
        """
        lane_left = frame_width * (0.5 - self.ego_lane_ratio / 2)
        lane_right = frame_width * (0.5 + self.ego_lane_ratio / 2)

        seen_ids = set()
        best_ttc: Optional[float] = None
        best_id: Optional[int] = None

        for obj in tracked_objects:
            if obj.class_name not in VEHICLE_CLASSES:
                continue

            seen_ids.add(obj.tracking_id)
            history = self._history.setdefault(obj.tracking_id, deque())
            history.append((timestamp, obj.height()))

            cutoff = timestamp - self.history_window
            while history and history[0][0] < cutoff:
                history.popleft()

            if not (lane_left <= obj.center_x <= lane_right):
                continue

            growth_rate = self._estimate_growth_rate(history)
            if growth_rate is None or growth_rate < self.min_growth_rate:
                continue  # not approaching (or not fast enough to matter)

            current_height = obj.height()
            if current_height <= 0:
                continue

            ttc = current_height / growth_rate
            if ttc < 0:
                continue

            if best_ttc is None or ttc < best_ttc:
                best_ttc = ttc
                best_id = obj.tracking_id

        # Prune history for objects no longer tracked, to avoid unbounded growth
        stale_ids = set(self._history.keys()) - seen_ids
        for stale_id in stale_ids:
            del self._history[stale_id]

        if best_ttc is None:
            return NearMissResult(near_miss_level=NEAR_MISS_NONE)

        if best_ttc <= self.critical_ttc:
            level = NEAR_MISS_CRITICAL
        elif best_ttc <= self.warning_ttc:
            level = NEAR_MISS_WARNING
        else:
            level = NEAR_MISS_NONE

        return NearMissResult(
            near_miss_level=level,
            time_to_collision_seconds=best_ttc,
            tracking_id=best_id,
        )

    def reset(self) -> None:
        """Reset all per-object history (e.g., starting a new video)."""
        self._history = {}
