"""
Near-miss panel component — Phase 23.

Displays the current time-to-collision / near-miss assessment for the
vehicle ahead, when applicable.
"""

from typing import Any, Dict, Optional

import streamlit as st


def render_near_miss(near_miss: Optional[Dict[str, Any]]) -> None:
    """
    Render the near-miss / TTC panel.

    Args:
        near_miss: dict matching NearMissResult.to_dict() shape
    """
    if not near_miss:
        return

    level = near_miss.get("near_miss_level", "NONE")
    ttc = near_miss.get("time_to_collision_seconds")

    if level == "NONE":
        return

    ttc_text = f" — est. {ttc:.1f}s to contact" if ttc is not None else ""

    if level == "CRITICAL":
        st.error(f"🚨 Near-miss warning{ttc_text}")
    elif level == "WARNING":
        st.warning(f"⚠️ Near-miss caution{ttc_text}")
