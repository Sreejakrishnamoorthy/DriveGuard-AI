"""
Incidents page: browse auto-recorded incidents — Phase 27.

Reads from the FastAPI backend, which reads incident manifests from
disk (via IncidentRecorder). Requires the backend to be running;
degrades gracefully with a clear message if it isn't.
"""

import os
import requests
import streamlit as st

BACKEND_URL = os.getenv("DRIVEGUARD_BACKEND_URL", "http://localhost:8000")


def _fetch_incidents(limit: int = 50):
    """Fetch recent incidents from the backend. Returns (incidents, error_message)."""
    try:
        response = requests.get(f"{BACKEND_URL}/api/incidents", params={"limit": limit}, timeout=5)
        response.raise_for_status()
        data = response.json()
        return data.get("incidents", []), None
    except requests.exceptions.ConnectionError:
        return [], f"Backend not reachable at {BACKEND_URL}. Start it with: uvicorn backend.main:app --reload"
    except requests.exceptions.Timeout:
        return [], "Backend request timed out."
    except Exception as e:
        return [], f"Error fetching incidents: {e}"


def render() -> None:
    """Render the incidents page."""
    st.markdown("### Incident Reconstruction")
    st.caption(
        "Automatically saved whenever a HIGH or CRITICAL severity alert fires "
        "(drowsiness, near-miss, unsafe distance, or overall high risk), capturing "
        "the annotated frame(s) and full risk breakdown at that moment. "
        "Requires the backend to be running (`uvicorn backend.main:app --reload`)."
    )

    if st.button("Refresh", key="incidents_refresh"):
        st.session_state.pop("incidents_list", None)

    if "incidents_list" not in st.session_state:
        incidents, error = _fetch_incidents()
        st.session_state.incidents_list = incidents
        st.session_state.incidents_error = error

    incidents = st.session_state.incidents_list
    error = st.session_state.incidents_error

    if error:
        st.warning(error)
        return

    if not incidents:
        st.info("No incidents recorded yet. Incidents are created automatically when a video processed via the API triggers a high-severity alert.")
        return

    for incident in incidents:
        severity = incident.get("severity", "")
        icon = "🔴" if severity == "CRITICAL" else "🟠"
        with st.expander(
            f"{icon} {incident.get('alert_type', '')} — {incident.get('message', '')} "
            f"— {incident.get('timestamp', '')}"
        ):
            col1, col2 = st.columns(2)
            col1.metric("Risk Score", f"{incident.get('risk_score', 0)}/100")
            col2.metric("Risk Level", incident.get("risk_level", ""))

            breakdown = incident.get("factor_breakdown", [])
            if breakdown:
                st.markdown("**Contributing factors:**")
                for f in breakdown:
                    st.write(f"+{f.get('points')} — {f.get('description')}")

            road_frame_path = incident.get("road_frame_path")
            driver_frame_path = incident.get("driver_frame_path")
            col3, col4 = st.columns(2)
            with col3:
                if road_frame_path and os.path.exists(road_frame_path):
                    st.image(road_frame_path, caption="Road camera at incident", width='stretch')
            with col4:
                if driver_frame_path and os.path.exists(driver_frame_path):
                    st.image(driver_frame_path, caption="Driver camera at incident", width='stretch')
