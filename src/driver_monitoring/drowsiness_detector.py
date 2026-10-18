"""
Drowsiness Detector for Driver Monitoring.

Consumes per-frame eye state from EyeDetector over TIME and determines
drowsiness — deliberately NOT from a single frame. A single closed-eye
frame is a blink; drowsiness is about sustained or repeated closure.

Uses two complementary temporal signals, similar in spirit to PERCLOS
(percentage of eyelid closure over time), a metric used in real-world
driver-drowsiness research:

    1. Consecutive closed-eye duration: how long eyes have been
       continuously closed right now (catches "nodding off").
    2. Rolling-window closure ratio: fraction of recent time spent with
       eyes closed (catches frequent/heavy blinking or fluttering that
       never quite reaches one long continuous closure).
"""

from collections import deque
from dataclasses import dataclass
from typing import Deque, Tuple
import time

from src.driver_monitoring.eye_detector import EYE_STATE_CLOSED, EYE_STATE_OPEN
from src.utils.logger import logger


@dataclass
class DrowsinessResult:
    """Result of temporal drowsiness analysis."""

    eye_state: str
    drowsiness_probability: float
    drowsiness: bool
    consecutive_closed_seconds: float = 0.0
    perclos: float = 0.0

    def to_dict(self) -> dict:
        return {
            "eye_state": self.eye_state,
            "drowsiness_probability": round(self.drowsiness_probability, 3),
            "drowsiness": self.drowsiness,
        }


class DrowsinessDetector:
    """
    Tracks eye state over time to determine drowsiness, requiring
    prolonged or repeated eye closure rather than flagging on one frame.
    """

    def __init__(
        self,
        closed_duration_threshold_seconds: float = 1.5,
        perclos_window_seconds: float = 10.0,
        perclos_threshold: float = 0.4,
    ) -> None:
        """
        Initialize drowsiness detector.

        Args:
            closed_duration_threshold_seconds: If eyes are continuously
                closed for at least this long, drowsiness triggers
                immediately (independent of the rolling window).
            perclos_window_seconds: Size of the rolling time window used
                to compute the closed-eye ratio (PERCLOS-style).
            perclos_threshold: Fraction of the rolling window (0-1) that
                must be closed-eye to trigger drowsiness via this signal.
        """
        self.closed_duration_threshold = closed_duration_threshold_seconds
        self.perclos_window = perclos_window_seconds
        self.perclos_threshold = perclos_threshold

        # (timestamp, is_closed) pairs within the rolling window
        self._history: Deque[Tuple[float, bool]] = deque()
        self._closed_since: float = None  # timestamp when current closure started

    def update(self, eye_state: str, timestamp: float = None) -> DrowsinessResult:
        """
        Update temporal state with the latest per-frame eye state.

        Args:
            eye_state: "OPEN", "CLOSED", or "UNKNOWN" from EyeDetector
            timestamp: Unix timestamp for this frame. Defaults to wall-clock
                time; pass explicit timestamps when processing a video file
                faster/slower than real-time so drowsiness timing reflects
                the VIDEO's timeline, not wall-clock processing speed.

        Returns:
            DrowsinessResult with current drowsiness assessment
        """
        if timestamp is None:
            timestamp = time.time()

        is_closed = eye_state == EYE_STATE_CLOSED

        # UNKNOWN (no face/landmarks) neither starts nor extends a closure,
        # and is excluded from the rolling window so missing-face frames
        # don't artificially dilute or inflate PERCLOS.
        if eye_state != "UNKNOWN":
            self._history.append((timestamp, is_closed))

            if is_closed:
                if self._closed_since is None:
                    self._closed_since = timestamp
            else:
                self._closed_since = None

        # Drop entries outside the rolling window
        cutoff = timestamp - self.perclos_window
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()

        consecutive_closed = (
            (timestamp - self._closed_since) if self._closed_since is not None else 0.0
        )

        perclos = 0.0
        history_span = 0.0
        if self._history:
            closed_count = sum(1 for _, closed in self._history if closed)
            perclos = closed_count / len(self._history)
            history_span = self._history[-1][0] - self._history[0][0]

        # PERCLOS is a "fraction of a full window" metric — it is only
        # statistically meaningful once we've actually observed close to a
        # full window of data. Without this guard, a handful of early
        # CLOSED frames (e.g. 1 second of data, 100% closed) would look
        # identical to a genuinely sustained 100%-closed window and could
        # trigger drowsiness almost instantly on startup.
        history_is_sufficient = history_span >= (self.perclos_window * 0.8)

        # Combine both signals into a single probability estimate.
        # Each signal is normalized to 0-1 against its own trigger
        # threshold, and we take the max (either signal alone can
        # indicate drowsiness; they don't need to agree).
        duration_score = min(consecutive_closed / self.closed_duration_threshold, 1.0) \
            if self.closed_duration_threshold > 0 else 0.0
        perclos_score = (
            min(perclos / self.perclos_threshold, 1.0)
            if self.perclos_threshold > 0 and history_is_sufficient
            else 0.0
        )

        drowsiness_probability = max(duration_score, perclos_score)
        drowsiness = (
            consecutive_closed >= self.closed_duration_threshold
            or (history_is_sufficient and perclos >= self.perclos_threshold)
        )

        return DrowsinessResult(
            eye_state=eye_state,
            drowsiness_probability=drowsiness_probability,
            drowsiness=drowsiness,
            consecutive_closed_seconds=consecutive_closed,
            perclos=perclos,
        )

    def reset(self) -> None:
        """Reset temporal state (e.g., when starting a new video/session)."""
        self._history.clear()
        self._closed_since = None
