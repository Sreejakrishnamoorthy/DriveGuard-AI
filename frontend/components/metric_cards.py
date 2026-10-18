"""
Metric card components: driver status and road analysis summaries.
"""

from typing import Any, Dict

import streamlit as st


def render_driver_metrics(driver_status: Dict[str, Any]) -> None:
    """
    Render driver-monitoring metric cards: drowsiness, attention, eye
    state, and overall driver status.

    Args:
        driver_status: dict matching DriverAnalysisResult.to_dict() shape
    """
    st.markdown("##### Driver Status")

    if not driver_status.get("face_detected"):
        st.warning("No face detected in driver camera.")
        return

    col1, col2, col3, col4 = st.columns(4)

    drowsy = driver_status.get("drowsiness", False)
    col1.metric("Drowsiness", "DETECTED" if drowsy else "Normal", delta=None)

    attention = driver_status.get("attention", "UNKNOWN")
    col2.metric("Attention", attention)

    eye_state = driver_status.get("eye_state", "UNKNOWN")
    col3.metric("Eye State", eye_state)

    status = driver_status.get("driver_status", "UNKNOWN")
    col4.metric("Overall Status", status)


def render_road_metrics(road_status: Dict[str, Any]) -> None:
    """
    Render road-analysis metric cards: vehicle counts, lane status,
    following distance, and hazards.

    Args:
        road_status: dict matching RoadAnalysisResult.to_dict() shape
    """
    st.markdown("##### Road Analysis")

    summary = road_status.get("vehicle_summary", {}) or {}
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Cars", summary.get("cars", 0))
    col2.metric("Bikes/Motorcycles", summary.get("bicycles", 0) + summary.get("motorcycles", 0))
    col3.metric("Pedestrians", summary.get("pedestrians", 0))
    col4.metric("Total Tracked", summary.get("total", 0))

    col5, col6, col7 = st.columns(3)

    lane = road_status.get("lane", {}) or {}
    lane_status = lane.get("lane_status", "UNKNOWN")
    col5.metric("Lane Status", lane_status)

    following_distance = road_status.get("following_distance", "SAFE")
    col6.metric("Following Distance", following_distance)

    hazard_count = len(road_status.get("hazards", []) or [])
    col7.metric("Hazards Detected", hazard_count)
