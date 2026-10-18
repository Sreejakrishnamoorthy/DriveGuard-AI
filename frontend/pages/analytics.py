"""
Analytics page: charts for the current session's risk trend and alert
patterns. Uses data from the current browser session (populated by the
Dashboard page after processing a video) since that's guaranteed
available without requiring the backend to be running.
"""

import streamlit as st

from frontend.components.charts import (
    render_risk_timeline_chart,
    render_alert_counts_chart,
    render_event_summary_charts,
)


def render() -> None:
    """Render the analytics page."""
    st.markdown("### Analytics")

    risk_history = st.session_state.get("risk_history", [])
    alert_history = st.session_state.get("alert_history", [])

    if not risk_history and not alert_history:
        st.info(
            "No analytics yet for this session. Go to the Dashboard tab, "
            "upload a video, and click START ANALYSIS first."
        )
        return

    render_event_summary_charts(alert_history)
    st.markdown("---")
    render_risk_timeline_chart(risk_history)
    st.markdown("---")
    render_alert_counts_chart(alert_history)
