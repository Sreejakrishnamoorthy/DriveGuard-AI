"""
DriveGuard AI - Streamlit Dashboard Entry Point.

Run with:
    streamlit run frontend/app.py
"""

import sys
from pathlib import Path

# Ensure project root is importable when Streamlit runs this file directly
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from frontend.pages import dashboard, history, analytics, incidents
from src.utils.config import Config

st.set_page_config(
    page_title="DriveGuard AI",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)


def render_header() -> None:
    """Render the app header."""
    st.markdown(
        """
        <div style="padding-bottom: 8px;">
            <h1 style="margin-bottom: 0;">🚗 DriveGuard AI</h1>
            <p style="color: #6b7280; font-size: 16px; margin-top: 4px;">
                Multimodal Real-Time Driver Risk &amp; Road Intelligence
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> str:
    """Render the sidebar navigation and device/status info. Returns the selected page name."""
    with st.sidebar:
        st.markdown("## Navigation")
        page = st.radio(
            "Go to",
            options=["Dashboard", "History", "Analytics", "Incidents"],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("## System Status")
        device = Config.validate_device()
        st.write(f"**Device:** `{device}`")
        st.write(f"**Model:** `{Config.MODEL_NAME}`")

        st.markdown("---")
        st.caption(
            "Road hazard detection uses an experimental OpenCV heuristic "
            "fallback unless a trained model is configured via "
            "`HAZARD_MODEL_PATH`. Lane detection uses a classical "
            "OpenCV Hough-transform approach."
        )

    return page


def main() -> None:
    render_header()
    page = render_sidebar()

    if page == "Dashboard":
        dashboard.render()
    elif page == "History":
        history.render()
    elif page == "Analytics":
        analytics.render()
    elif page == "Incidents":
        incidents.render()


if __name__ == "__main__":
    main()
