"""
Video processing utilities for DriveGuard AI.

Handles video loading, frame extraction, and video writing.
"""

import os
from typing import Optional, Tuple
import cv2
import numpy as np

from src.utils.logger import logger


class VideoProcessor:
    """Handles video loading and processing."""

    def __init__(self, video_path: str) -> None:
        """
        Initialize video processor.

        Args:
            video_path: Path to video file

        Raises:
            FileNotFoundError: If video file doesn't exist
            ValueError: If video cannot be opened
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self.video_path = video_path
        self.cap = None
        self.frame_count = 0
        self.fps = 0
        self.width = 0
        self.height = 0

        self._open_video()

    def _open_video(self) -> None:
        """Open video file and extract metadata."""
        try:
            self.cap = cv2.VideoCapture(self.video_path)

            if not self.cap.isOpened():
                raise ValueError(f"Cannot open video: {self.video_path}")

            # Extract video properties
            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            logger.info(f"Video loaded: {os.path.basename(self.video_path)}")
            logger.info(f"  Frames: {self.frame_count}")
            logger.info(f"  FPS: {self.fps}")
            logger.info(f"  Resolution: {self.width}x{self.height}")

        except Exception as e:
            logger.error(f"Error opening video: {e}")
            raise

    def get_frame_count(self) -> int:
        """Get total frame count."""
        return self.frame_count

    def get_fps(self) -> float:
        """Get frames per second."""
        return self.fps

    def get_resolution(self) -> Tuple[int, int]:
        """Get video resolution (width, height)."""
        return (self.width, self.height)

    def get_duration_seconds(self) -> float:
        """Get video duration in seconds."""
        if self.fps == 0:
            return 0
        return self.frame_count / self.fps

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read next frame from video.

        Returns:
            Tuple of (success, frame) where frame is numpy array or None
        """
        if self.cap is None:
            return False, None

        success, frame = self.cap.read()
        return success, frame

    def set_frame_position(self, frame_number: int) -> None:
        """
        Set video to specific frame.

        Args:
            frame_number: Frame number to seek to
        """
        if self.cap is None:
            logger.warning("Video not open")
            return

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        logger.debug(f"Seeked to frame {frame_number}")

    def get_current_frame_number(self) -> int:
        """Get current frame number."""
        if self.cap is None:
            return 0
        return int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

    def close(self) -> None:
        """Close video file."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            logger.debug(f"Video closed: {os.path.basename(self.video_path)}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __del__(self):
        """Destructor."""
        self.close()


class VideoWriter:
    """Handles video output writing."""

    def __init__(
        self,
        output_path: str,
        fps: float,
        frame_size: Tuple[int, int],
        codec: str = "mp4v",
    ) -> None:
        """
        Initialize video writer.

        Args:
            output_path: Path to output video file
            fps: Frames per second
            frame_size: Frame size (width, height)
            codec: Video codec (default: 'mp4v')
        """
        self.output_path = output_path
        self.fps = fps
        self.frame_size = frame_size

        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        # Initialize writer
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

        if not self.writer.isOpened():
            logger.error(f"Failed to open video writer: {output_path}")
            raise RuntimeError(f"Cannot open video writer for {output_path}")

        logger.info(f"Video writer initialized: {output_path}")
        logger.info(f"  Resolution: {frame_size[0]}x{frame_size[1]}")
        logger.info(f"  FPS: {fps}")

    def write_frame(self, frame: np.ndarray) -> None:
        """
        Write frame to video.

        Args:
            frame: Frame to write
        """
        if self.writer is not None:
            self.writer.write(frame)

    def close(self) -> None:
        """Close video writer."""
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            logger.info(f"Video written: {self.output_path}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __del__(self):
        """Destructor."""
        self.close()


def resize_frame(
    frame: np.ndarray,
    max_width: int = 640,
    max_height: int = 480,
) -> np.ndarray:
    """
    Resize frame maintaining aspect ratio.

    Args:
        frame: Input frame
        max_width: Maximum width
        max_height: Maximum height

    Returns:
        Resized frame
    """
    height, width = frame.shape[:2]

    # Calculate scaling factor
    scale = min(max_width / width, max_height / height)

    if scale < 1:
        new_width = int(width * scale)
        new_height = int(height * scale)
        frame = cv2.resize(frame, (new_width, new_height))

    return frame
