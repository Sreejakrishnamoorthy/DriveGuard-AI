"""
Temporal Analyzer for the Risk Engine.

Generic, reusable event-persistence and cooldown tracking, used by
AlertManager so that a condition which stays true for many consecutive
frames produces ONE alert (after a minimum persistence duration), not one
alert per frame.

Example (from project spec): if drowsiness remains true for 3 seconds,
trigger ONE drowsiness alert — not 90 alerts for 90 consecutive frames
at 30fps.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional
import time

from src.utils.logger import logger


@dataclass
class TemporalEventState:
    """Internal per-event-type temporal state."""

    active_since: Optional[float] = None  # when the current true-streak started
    last_fired_at: Optional[float] = None  # when an alert was last actually fired
    fire_history: Deque[float] = field(default_factory=deque)  # timestamps of past firings


@dataclass
class TemporalUpdateResult:
    """Result of updating temporal state for one event type on one frame."""

    event_type: str
    is_active: bool
    should_fire: bool  # True exactly on frames where a NEW alert should be raised
    duration_active_seconds: float = 0.0
    recent_fire_count: int = 0  # firings within the rolling count window


class TemporalAnalyzer:
    """
    Tracks how long each named event has been continuously active, and
    decides when a new alert should actually fire based on:

        1. Persistence: the event must stay active for at least
           `persistence_seconds` before it first fires (filters out
           single-frame noise/flicker).
        2. Cooldown: after firing, the SAME event type won't fire again
           for at least `cooldown_seconds`, even if it stays active the
           whole time (prevents one alert per frame for a sustained
           condition).

    Each event type is tracked independently, so "drowsiness" and
    "lane_violation" have separate persistence/cooldown clocks.
    """

    def __init__(
        self,
        persistence_seconds: float = 3.0,
        cooldown_seconds: float = 10.0,
        count_window_seconds: float = 300.0,
    ) -> None:
        """
        Initialize the temporal analyzer.

        Args:
            persistence_seconds: Minimum continuous-active duration before
                an event is allowed to fire at all.
            cooldown_seconds: Minimum time between successive firings of
                the SAME event type, even if it remains continuously active.
            count_window_seconds: Rolling window used for recent_fire_count
                (e.g. "3 drowsiness alerts in the last 5 minutes").
        """
        self.persistence_seconds = persistence_seconds
        self.cooldown_seconds = cooldown_seconds
        self.count_window_seconds = count_window_seconds

        self._states: Dict[str, TemporalEventState] = {}

    def update(
        self, event_type: str, is_active: bool, timestamp: float = None
    ) -> TemporalUpdateResult:
        """
        Update temporal state for one event type and determine whether a
        new alert should fire this frame.

        Args:
            event_type: Name of the event/condition (e.g. "DROWSINESS")
            is_active: Whether the underlying condition is true this frame
            timestamp: Explicit timestamp (seconds). Defaults to wall-clock
                time; pass explicit timestamps when processing a video file
                so persistence/cooldown reflect the VIDEO's timeline.

        Returns:
            TemporalUpdateResult — check `should_fire` to know whether to
            actually create/emit an alert this frame.
        """
        if timestamp is None:
            timestamp = time.time()

        state = self._states.setdefault(event_type, TemporalEventState())

        if not is_active:
            state.active_since = None
            return TemporalUpdateResult(
                event_type=event_type, is_active=False, should_fire=False
            )

        if state.active_since is None:
            state.active_since = timestamp

        duration_active = timestamp - state.active_since

        # Drop firing history outside the counting window
        cutoff = timestamp - self.count_window_seconds
        while state.fire_history and state.fire_history[0] < cutoff:
            state.fire_history.popleft()

        should_fire = False
        # Small epsilon guards against floating-point accumulation drift
        # when timestamps are built by repeatedly summing frame deltas
        # (e.g. t += 1/fps many times) — without it, a duration that is
        # mathematically exactly at the threshold can land a few
        # nanoseconds short and delay firing by one extra frame.
        if duration_active >= self.persistence_seconds - 1e-9:
            in_cooldown = (
                state.last_fired_at is not None
                and (timestamp - state.last_fired_at) < self.cooldown_seconds
            )
            if not in_cooldown:
                should_fire = True
                state.last_fired_at = timestamp
                state.fire_history.append(timestamp)

        return TemporalUpdateResult(
            event_type=event_type,
            is_active=True,
            should_fire=should_fire,
            duration_active_seconds=duration_active,
            recent_fire_count=len(state.fire_history),
        )

    def reset(self, event_type: Optional[str] = None) -> None:
        """
        Reset temporal state.

        Args:
            event_type: If given, reset only that event type. If None,
                reset all tracked event types (e.g. for a new session).
        """
        if event_type is None:
            self._states.clear()
            logger.info("TemporalAnalyzer: all event states reset.")
        elif event_type in self._states:
            del self._states[event_type]
