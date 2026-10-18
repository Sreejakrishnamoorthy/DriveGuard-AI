"""
Alert panel component: displays active/recent alerts, prioritized by
severity (Phase 20) so the most urgent concern is always shown first.
"""

from typing import Any, Dict, List, Optional

import streamlit as st

SEVERITY_ICONS = {
    "LOW": "ℹ️",
    "MEDIUM": "⚠️",
    "HIGH": "⚠️",
    "CRITICAL": "🚨",
}

# Mirrors src/risk_engine/alert_manager.py's SEVERITY_RANK. Kept as a
# plain, pure-function-friendly mapping here since the frontend receives
# alerts as plain dicts (via session state or the API), not AlertManager
# instances.
SEVERITY_RANK = {"CRITICAL": 3, "HIGH": 2, "MEDIUM": 1, "LOW": 0}


def get_severity_icon(severity: str) -> str:
    """Map an alert severity to a display icon. Pure function, easily testable."""
    return SEVERITY_ICONS.get(severity, "⚠️")


def sort_alerts_by_priority(alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Phase 20 — Alert Prioritization.

    Sort alerts by severity (most urgent first), using original list
    order as a recency tiebreaker within the same severity. Pure
    function, easily testable independent of Streamlit.
    """
    indexed = list(enumerate(alerts))
    ranked = sorted(
        indexed,
        key=lambda pair: (SEVERITY_RANK.get(pair[1].get("severity", ""), -1), pair[0]),
        reverse=True,
    )
    return [alert for _, alert in ranked]


def get_top_priority_alert(alerts: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The single most urgent alert in the list, or None if empty."""
    prioritized = sort_alerts_by_priority(alerts)
    return prioritized[0] if prioritized else None


def render_alerts(alerts: List[Dict[str, Any]], prioritize: bool = True) -> None:
    """
    Render the active-alerts panel.

    Args:
        alerts: list of dicts matching Alert.to_dict() shape (type, severity, message, timestamp)
        prioritize: if True (default), sort by severity so the most
            urgent alert appears first, with a highlighted callout banner
            for the single top-priority alert. If False, falls back to
            simple recency ordering (most recent first).
    """
    st.markdown("##### Active Alerts")

    if not alerts:
        st.success("No active alerts.")
        return

    ordered = sort_alerts_by_priority(alerts) if prioritize else list(reversed(alerts))

    if prioritize:
        top = ordered[0]
        icon = get_severity_icon(top.get("severity", ""))
        st.markdown(f"**{icon} Most urgent: {top.get('message', '')}**")
        st.markdown("---")

    for alert in ordered:
        icon = get_severity_icon(alert.get("severity", "MEDIUM"))
        severity = alert.get("severity", "")
        message = alert.get("message", "")
        timestamp = alert.get("timestamp", "")

        if severity == "CRITICAL":
            st.error(f"{icon} **{message}** ({timestamp})")
        elif severity == "HIGH":
            st.warning(f"{icon} **{message}** ({timestamp})")
        else:
            st.info(f"{icon} {message} ({timestamp})")
