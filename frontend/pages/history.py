"""
History page: lists past analysis sessions, read from the FastAPI backend
(which itself reads from MongoDB). Requires the backend to be running;
degrades gracefully with a clear message if it isn't.
"""

import os
import requests
import streamlit as st

BACKEND_URL = os.getenv("DRIVEGUARD_BACKEND_URL", "http://localhost:8000")


def _fetch_sessions(limit: int = 50):
    """Fetch recent sessions from the backend. Returns (sessions, error_message)."""
    try:
        response = requests.get(f"{BACKEND_URL}/api/sessions", params={"limit": limit}, timeout=5)
        response.raise_for_status()
        data = response.json()
        return data.get("sessions", []), None
    except requests.exceptions.ConnectionError:
        return [], f"Backend not reachable at {BACKEND_URL}. Start it with: uvicorn backend.main:app --reload"
    except requests.exceptions.Timeout:
        return [], "Backend request timed out."
    except Exception as e:
        return [], f"Error fetching sessions: {e}"


def render() -> None:
    """Render the session history page."""
    st.markdown("### Session History")
    st.caption(
        "Past analysis sessions, persisted via the FastAPI backend and MongoDB. "
        "Requires the backend to be running separately (`uvicorn backend.main:app --reload`)."
    )

    if st.button("Refresh"):
        st.session_state.pop("history_sessions", None)

    if "history_sessions" not in st.session_state:
        sessions, error = _fetch_sessions()
        st.session_state.history_sessions = sessions
        st.session_state.history_error = error

    sessions = st.session_state.history_sessions
    error = st.session_state.history_error

    if error:
        st.warning(error)
        return

    if not sessions:
        st.info("No sessions recorded yet. Sessions are created when you process a video via the API.")
        return

    for session in sessions:
        risk_level = session.get("risk_level", "LOW")
        icon = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🟠", "CRITICAL": "🔴"}.get(risk_level, "⚪")
        with st.expander(
            f"{icon} Session {session.get('session_id', '')[:8]}... "
            f"— {risk_level} (score {session.get('risk_score', 0)}) "
            f"— {session.get('start_time', '')}"
        ):
            col1, col2 = st.columns(2)
            col1.metric("Driver Status", session.get("driver_status", "UNKNOWN"))
            col2.metric("Road Status", session.get("road_status", "UNKNOWN"))
            st.write(f"**Start:** {session.get('start_time', '-')}")
            st.write(f"**End:** {session.get('end_time') or 'In progress'}")
            alerts = session.get("alerts", [])
            st.write(f"**Alerts during session:** {len(alerts)}")
            if alerts:
                st.json(alerts)
