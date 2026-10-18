"""
Analytics chart components: risk-over-time and alert-frequency visualizations.

Data-preparation logic is kept in plain functions (no Streamlit/Plotly
calls) so it can be unit tested directly; render_* functions just feed
that data into a chart.
"""

from collections import Counter
from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def prepare_risk_timeline(risk_events: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert a list of risk-event dicts (timestamp, risk_score, risk_level)
    into a DataFrame sorted by time, ready for charting.

    Args:
        risk_events: list of dicts with at least "timestamp" and "risk_score"

    Returns:
        DataFrame with columns [timestamp, risk_score, risk_level]
    """
    if not risk_events:
        return pd.DataFrame(columns=["timestamp", "risk_score", "risk_level"])

    df = pd.DataFrame(risk_events)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp")
    return df


def prepare_alert_type_counts(alerts: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Count alerts by type.

    Args:
        alerts: list of alert dicts with a "type" field

    Returns:
        DataFrame with columns [type, count], sorted descending by count
    """
    if not alerts:
        return pd.DataFrame(columns=["type", "count"])

    counts = Counter(a.get("type", "UNKNOWN") for a in alerts)
    df = pd.DataFrame(sorted(counts.items(), key=lambda x: x[1], reverse=True), columns=["type", "count"])
    return df


def count_alerts_by_type(alerts: List[Dict[str, Any]], alert_type: str) -> int:
    """Count how many alerts of a specific type occurred. Pure function, easily testable."""
    return sum(1 for a in alerts if a.get("type") == alert_type)


def render_risk_timeline_chart(risk_events: List[Dict[str, Any]]) -> None:
    """Render risk score over time as a line chart."""
    df = prepare_risk_timeline(risk_events)

    if df.empty:
        st.info("No risk history yet.")
        return

    fig = px.line(
        df,
        x="timestamp",
        y="risk_score",
        title="Risk Score Over Time",
        markers=True,
    )
    fig.update_yaxes(range=[0, 100], title="Risk Score")
    fig.update_xaxes(title="Time")
    fig.add_hrect(y0=0, y1=30, fillcolor="#22c55e", opacity=0.08, line_width=0)
    fig.add_hrect(y0=31, y1=60, fillcolor="#eab308", opacity=0.08, line_width=0)
    fig.add_hrect(y0=61, y1=80, fillcolor="#f97316", opacity=0.08, line_width=0)
    fig.add_hrect(y0=81, y1=100, fillcolor="#ef4444", opacity=0.08, line_width=0)
    st.plotly_chart(fig, width='stretch')


def render_alert_counts_chart(alerts: List[Dict[str, Any]]) -> None:
    """Render alert counts by type as a bar chart."""
    df = prepare_alert_type_counts(alerts)

    if df.empty:
        st.info("No alerts recorded yet.")
        return

    fig = px.bar(df, x="type", y="count", title="Alerts by Type", color="type")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, width='stretch')


def render_event_summary_charts(alerts: List[Dict[str, Any]]) -> None:
    """Render a small multi-metric summary: drowsiness, lane violations, hazards."""
    st.markdown("##### Event Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Drowsiness Events", count_alerts_by_type(alerts, "DROWSINESS"))
    col2.metric("Lane Violations", count_alerts_by_type(alerts, "LANE_DEPARTURE"))
    col3.metric("Road Hazards", count_alerts_by_type(alerts, "ROAD_HAZARD"))
