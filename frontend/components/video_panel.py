"""
Video panel component: displays a road or driver camera frame with a label.
"""

from typing import Optional

import numpy as np
import streamlit as st


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    """Convert an OpenCV BGR frame to RGB for Streamlit display. Pure function, easily testable."""
    return frame[:, :, ::-1]


def render_video_panel(title: str, frame: Optional[np.ndarray] = None, caption: str = "") -> None:
    """
    Render a labeled video panel showing either the given annotated frame
    or a placeholder message if no frame is available yet.

    Args:
        title: Panel heading (e.g. "ROAD CAMERA")
        frame: Annotated BGR frame (numpy array) to display, or None
        caption: Optional caption text under the frame
    """
    st.markdown(f"**{title}**")
    if frame is not None:
        st.image(bgr_to_rgb(frame), width='stretch', caption=caption or None)
    else:
        st.info(f"No {title.lower()} feed yet — upload a video to begin.")
