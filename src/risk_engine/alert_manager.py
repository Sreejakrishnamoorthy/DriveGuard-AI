"""
Alert Manager.

Generates structured alerts from risk factors, using TemporalAnalyzer so
a sustained condition (e.g. drowsiness lasting 10 seconds) produces one
alert plus periodic re-alerts after cooldown — not one alert per frame.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional
import time

from src.risk_engine.risk_calculator import RiskFactors, RiskScoreResult
from src.risk_engine.temporal_analyzer import TemporalAnalyzer
from src.utils.logger import logger

ALERT_TYPE_DROWSINESS = "DROWSINESS"
ALERT_TYPE_DISTRACTION = "DISTRACTION"
ALERT_TYPE_LANE_DEPARTURE = "LANE_DEPARTURE"
ALERT_TYPE_UNSAFE_DISTANCE = "UNSAFE_DISTANCE"
ALERT_TYPE_ROAD_HAZARD = "ROAD_HAZARD"
ALERT_TYPE_HIGH_RISK = "HIGH_RISK"
ALERT_TYPE_NEAR_MISS = "NEAR_MISS"

SEVERITY_LOW = "LOW"
SEVERITY_MEDIUM = "MEDIUM"
SEVERITY_HIGH = "HIGH"
SEVERITY_CRITICAL = "CRITICAL"

# Phase 20 — Alert Prioritization. Higher number = more urgent. Used to
# rank a batch of simultaneously-active alerts so the driver sees the
# single most important thing first, instead of a flat, unordered list.
SEVERITY_RANK = {
    SEVERITY_CRITICAL: 3,
    SEVERITY_HIGH: 2,
    SEVERITY_MEDIUM: 1,
    SEVERITY_LOW: 0,
}

# Default severity + human-readable message per alert type. NEAR_MISS's
# message is generated dynamically (includes the TTC estimate) rather
# than using this static default — see process().
_ALERT_DEFAULTS: Dict[str, Dict[str, str]] = {
    ALERT_TYPE_DROWSINESS: {"severity": SEVERITY_HIGH, "message": "Driver drowsiness detected"},
    ALERT_TYPE_DISTRACTION: {"severity": SEVERITY_MEDIUM, "message": "Driver distraction detected"},
    ALERT_TYPE_LANE_DEPARTURE: {"severity": SEVERITY_MEDIUM, "message": "Lane departure detected"},
    ALERT_TYPE_UNSAFE_DISTANCE: {"severity": SEVERITY_HIGH, "message": "Unsafe following distance"},
    ALERT_TYPE_ROAD_HAZARD: {"severity": SEVERITY_MEDIUM, "message": "Road hazard detected ahead"},
    ALERT_TYPE_HIGH_RISK: {"severity": SEVERITY_HIGH, "message": "Overall driving risk is elevated"},
    ALERT_TYPE_NEAR_MISS: {"severity": SEVERITY_CRITICAL, "message": "Near-miss warning"},
}


@dataclass
class Alert:
    """A single structured alert."""

    type: str
    severity: str
    message: str
    timestamp: str  # ISO 8601

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "severity": self.severity,
            "message": self.message,
            "timestamp": self.timestamp,
        }


def _iso_timestamp(epoch_seconds: Optional[float] = None) -> str:
    """Convert a Unix timestamp (or now) to an ISO 8601 UTC string."""
    ts = epoch_seconds if epoch_seconds is not None else time.time()
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


class AlertManager:
    """
    Converts risk factors into structured, temporally-debounced alerts.

    Wraps a TemporalAnalyzer internally so callers just call `process()`
    once per frame with the current risk factors/result — persistence and
    cooldown are handled automatically per alert type.
    """

    def __init__(
        self,
        persistence_seconds: float = 3.0,
        cooldown_seconds: float = 10.0,
        high_risk_persistence_seconds: float = 2.0,
        high_risk_cooldown_seconds: float = 15.0,
        near_miss_cooldown_seconds: float = 5.0,
        max_history: int = 200,
    ) -> None:
        """
        Initialize the alert manager.

        Args:
            persistence_seconds: How long a per-factor condition (drowsiness,
                distraction, etc.) must stay active before it first alerts.
            cooldown_seconds: Minimum time between repeat alerts of the SAME type.
            high_risk_persistence_seconds: Persistence specifically for the
                HIGH_RISK meta-alert (overall risk level, not a single factor).
            high_risk_cooldown_seconds: Cooldown specifically for HIGH_RISK.
            near_miss_cooldown_seconds: Cooldown between NEAR_MISS alerts.
                Unlike other alert types, near-miss has NO persistence
                requirement (fires on the very first critical/warning
                frame) since it represents an acute, fast-moving danger —
                waiting 3 seconds to confirm it, the way drowsiness does,
                would mean the alert arrives after the danger has passed.
                A shorter cooldown than other alerts is used for the same
                reason: near-miss situations can recur quickly in traffic.
            max_history: Maximum number of recent alerts kept in memory.
        """
        self._factor_temporal = TemporalAnalyzer(
            persistence_seconds=persistence_seconds, cooldown_seconds=cooldown_seconds
        )
        self._risk_temporal = TemporalAnalyzer(
            persistence_seconds=high_risk_persistence_seconds,
            cooldown_seconds=high_risk_cooldown_seconds,
        )
        self._near_miss_temporal = TemporalAnalyzer(
            persistence_seconds=0.0, cooldown_seconds=near_miss_cooldown_seconds
        )
        self.max_history = max_history
        self._history: List[Alert] = []

        logger.info("AlertManager initialized.")

    def _make_alert(self, alert_type: str, timestamp: float, message_override: Optional[str] = None) -> Alert:
        defaults = _ALERT_DEFAULTS[alert_type]
        return Alert(
            type=alert_type,
            severity=defaults["severity"],
            message=message_override or defaults["message"],
            timestamp=_iso_timestamp(timestamp),
        )

    def process(
        self,
        factors: RiskFactors,
        risk_result: RiskScoreResult,
        timestamp: Optional[float] = None,
        near_miss_level: Optional[str] = None,
        near_miss_ttc_seconds: Optional[float] = None,
    ) -> List[Alert]:
        """
        Evaluate current risk factors and risk result, returning any NEW
        alerts that should fire this frame (respecting persistence/cooldown).

        Args:
            factors: Current RiskFactors (from RiskCalculator input)
            risk_result: Current RiskScoreResult (from RiskCalculator.calculate())
            timestamp: Explicit timestamp (seconds). Defaults to wall-clock time.
            near_miss_level: Optional "NONE"/"WARNING"/"CRITICAL" from
                NearMissDetector (Phase 23). If provided and not "NONE",
                a NEAR_MISS alert is evaluated (immediate, no persistence
                delay — see __init__ docstring).
            near_miss_ttc_seconds: Estimated time-to-collision, included
                in the alert message when available.

        Returns:
            List of newly-fired Alert objects (often empty — most frames
            produce no new alert even when a condition is actively true,
            by design).
        """
        if timestamp is None:
            timestamp = time.time()

        new_alerts: List[Alert] = []

        factor_checks = [
            (ALERT_TYPE_DROWSINESS, factors.drowsiness),
            (ALERT_TYPE_DISTRACTION, factors.distraction),
            (ALERT_TYPE_LANE_DEPARTURE, factors.lane_violation),
            (ALERT_TYPE_UNSAFE_DISTANCE, factors.unsafe_following_distance),
            (ALERT_TYPE_ROAD_HAZARD, factors.road_hazard),
        ]

        for alert_type, is_active in factor_checks:
            result = self._factor_temporal.update(alert_type, is_active, timestamp)
            if result.should_fire:
                alert = self._make_alert(alert_type, timestamp)
                new_alerts.append(alert)
                logger.info(f"Alert fired: {alert.type} ({alert.severity}) - {alert.message}")

        # HIGH_RISK is a meta-alert on the overall score, independent of
        # which specific factors are driving it.
        is_high_risk = risk_result.risk_level in ("HIGH", "CRITICAL")
        risk_result_check = self._risk_temporal.update(ALERT_TYPE_HIGH_RISK, is_high_risk, timestamp)
        if risk_result_check.should_fire:
            severity = SEVERITY_CRITICAL if risk_result.risk_level == "CRITICAL" else SEVERITY_HIGH
            message = f"{risk_result.risk_level} driving risk detected (score {risk_result.risk_score}/100)"
            alert = Alert(
                type=ALERT_TYPE_HIGH_RISK,
                severity=severity,
                message=message,
                timestamp=_iso_timestamp(timestamp),
            )
            new_alerts.append(alert)
            logger.info(f"Alert fired: {alert.type} ({alert.severity}) - {alert.message}")

        # NEAR_MISS (Phase 23) — fires immediately (no persistence delay),
        # since a real near-miss is over in a second or two.
        is_near_miss = near_miss_level in ("WARNING", "CRITICAL")
        near_miss_check = self._near_miss_temporal.update(
            ALERT_TYPE_NEAR_MISS, is_near_miss, timestamp
        )
        if near_miss_check.should_fire:
            severity = SEVERITY_CRITICAL if near_miss_level == "CRITICAL" else SEVERITY_HIGH
            ttc_text = f", est. {near_miss_ttc_seconds:.1f}s to contact" if near_miss_ttc_seconds else ""
            message = f"Near-miss warning{ttc_text}"
            alert = Alert(
                type=ALERT_TYPE_NEAR_MISS,
                severity=severity,
                message=message,
                timestamp=_iso_timestamp(timestamp),
            )
            new_alerts.append(alert)
            logger.info(f"Alert fired: {alert.type} ({alert.severity}) - {alert.message}")

        for alert in new_alerts:
            self._history.append(alert)
        if len(self._history) > self.max_history:
            self._history = self._history[-self.max_history :]

        return new_alerts

    def get_recent_alerts(self, limit: Optional[int] = None) -> List[Alert]:
        """Get recent alerts, most recent last. Optionally limit count."""
        if limit is None:
            return list(self._history)
        return self._history[-limit:]

    def get_prioritized_alerts(self, limit: Optional[int] = None) -> List[Alert]:
        """
        Phase 20 — Alert Prioritization.

        Get recent alerts ranked by severity (most urgent first), with
        recency as a tiebreaker within the same severity. Use this instead
        of get_recent_alerts() when the display should surface the single
        most important active concern first, rather than just "what
        happened most recently."
        """
        alerts = self.get_recent_alerts()
        ranked = sorted(
            enumerate(alerts),
            key=lambda pair: (SEVERITY_RANK.get(pair[1].severity, -1), pair[0]),
            reverse=True,
        )
        ordered = [alert for _, alert in ranked]
        if limit is not None:
            return ordered[:limit]
        return ordered

    def get_top_priority_alert(self) -> Optional[Alert]:
        """
        Phase 20 — the single most urgent currently-recorded alert, or
        None if there are no alerts. Convenience wrapper around
        get_prioritized_alerts() for callers that just need "the one
        thing to show/announce right now" (e.g. voice alerts, Phase 19).
        """
        prioritized = self.get_prioritized_alerts(limit=1)
        return prioritized[0] if prioritized else None

    def clear_history(self) -> None:
        """Clear alert history (e.g., when starting a new session)."""
        self._history = []

    def reset(self) -> None:
        """Reset all temporal state and history (e.g., new video/session)."""
        self._factor_temporal.reset()
        self._risk_temporal.reset()
        self._near_miss_temporal.reset()
        self.clear_history()
        logger.info("AlertManager state reset.")
