"""
Inactivity Monitoring Engine
AI-Based Real-Time Livestock Behavior & Inactivity Monitoring System
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from config import (
    BEHAVIOR_STANDING,
    BEHAVIOR_FEEDING,
    BEHAVIOR_LYING,
    BEHAVIOR_UNKNOWN,
    DEFAULT_INACTIVITY_THRESHOLD_SECONDS,
    DEFAULT_TRACKING_GRACE_PERIOD_SECONDS,
    WARNING_THRESHOLD_RATIO,
)

logger = logging.getLogger(__name__)


@dataclass
class AnimalActivityState:
    """Maintains continuous tracking and activity statistics for a specific cow."""
    tracking_id: int
    first_detected_time: float
    last_detected_time: float
    last_frame_number: int
    current_behavior: str = BEHAVIOR_UNKNOWN
    current_behavior_start_time: float = 0.0
    current_lying_duration: float = 0.0
    total_observed_duration: float = 0.0
    feeding_duration: float = 0.0
    standing_duration: float = 0.0
    lying_duration: float = 0.0
    unknown_duration: float = 0.0
    status: str = "Active"  # "Active", "GracePeriod", "Lost"
    alert_severity: str = "Normal"  # "Normal", "Warning", "Critical"

    def get_behavior_durations(self) -> Dict[str, float]:
        """Returns duration breakdown by behavior in seconds."""
        return {
            BEHAVIOR_STANDING: round(self.standing_duration, 2),
            BEHAVIOR_FEEDING: round(self.feeding_duration, 2),
            BEHAVIOR_LYING: round(self.lying_duration, 2),
            BEHAVIOR_UNKNOWN: round(self.unknown_duration, 2),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tracking_id": self.tracking_id,
            "current_behavior": self.current_behavior,
            "status": self.status,
            "alert_severity": self.alert_severity,
            "current_lying_duration": round(self.current_lying_duration, 2),
            "total_observed_duration": round(self.total_observed_duration, 2),
            "feeding_duration": round(self.feeding_duration, 2),
            "standing_duration": round(self.standing_duration, 2),
            "lying_duration": round(self.lying_duration, 2),
            "unknown_duration": round(self.unknown_duration, 2),
            "first_detected_time": round(self.first_detected_time, 2),
            "last_detected_time": round(self.last_detected_time, 2),
        }


class InactivityEngine:
    """
    Tracks temporal behavior transitions, calculates continuous lying durations,
    and enforces grace periods when tracking is momentarily lost.
    """

    def __init__(
        self,
        inactivity_threshold_seconds: float = DEFAULT_INACTIVITY_THRESHOLD_SECONDS,
        grace_period_seconds: float = DEFAULT_TRACKING_GRACE_PERIOD_SECONDS,
    ):
        self.inactivity_threshold_seconds = inactivity_threshold_seconds
        self.grace_period_seconds = grace_period_seconds
        self.animals: Dict[int, AnimalActivityState] = {}

    def set_inactivity_threshold(self, seconds: float) -> None:
        """Dynamically updates the inactivity threshold."""
        if seconds > 0:
            self.inactivity_threshold_seconds = float(seconds)

    def reset(self) -> None:
        """Resets engine state for a new video session."""
        self.animals.clear()

    def update(
        self,
        active_detections: List[Dict[str, Any]],
        current_timestamp: float,
        frame_number: int,
    ) -> Tuple[List[AnimalActivityState], List[Dict[str, Any]]]:
        """
        Updates activity states for all tracked animals at current_timestamp.
        active_detections is a list of dicts with 'tracking_id' and 'behavior'.
        Returns (all_animal_states, completed_behavior_events_for_db).
        """
        completed_events: List[Dict[str, Any]] = []
        seen_ids = set()

        for det in active_detections:
            track_id = det["tracking_id"]
            new_behavior = det.get("behavior", BEHAVIOR_UNKNOWN)
            seen_ids.add(track_id)

            if track_id not in self.animals:
                # First time seeing this animal
                state = AnimalActivityState(
                    tracking_id=track_id,
                    first_detected_time=current_timestamp,
                    last_detected_time=current_timestamp,
                    last_frame_number=frame_number,
                    current_behavior=new_behavior,
                    current_behavior_start_time=current_timestamp,
                    status="Active",
                )
                self.animals[track_id] = state
            else:
                state = self.animals[track_id]
                state.status = "Active"
                state.last_frame_number = frame_number

                dt = max(0.0, current_timestamp - state.last_detected_time)

                # Behavior transition check
                if new_behavior != state.current_behavior:
                    # Previous behavior interval completed
                    event_duration = max(0.0, current_timestamp - state.current_behavior_start_time)
                    if event_duration > 0:
                        completed_events.append({
                            "tracking_id": track_id,
                            "behavior": state.current_behavior,
                            "duration_seconds": round(event_duration, 2),
                            "timestamp": current_timestamp,
                        })

                    # If transitioning away from Lying/Resting, reset continuous lying timer
                    if state.current_behavior == BEHAVIOR_LYING:
                        state.current_lying_duration = 0.0
                        state.alert_severity = "Normal"

                    state.current_behavior = new_behavior
                    state.current_behavior_start_time = current_timestamp
                else:
                    # Continuing same behavior
                    state.total_observed_duration += dt

                    if new_behavior == BEHAVIOR_STANDING:
                        state.standing_duration += dt
                    elif new_behavior == BEHAVIOR_FEEDING:
                        state.feeding_duration += dt
                    elif new_behavior == BEHAVIOR_LYING:
                        state.lying_duration += dt
                        state.current_lying_duration += dt
                    else:
                        state.unknown_duration += dt

                    # Evaluate severity thresholds
                    warning_threshold = self.inactivity_threshold_seconds * WARNING_THRESHOLD_RATIO
                    if state.current_lying_duration >= self.inactivity_threshold_seconds:
                        state.alert_severity = "Critical"
                    elif state.current_lying_duration >= warning_threshold:
                        state.alert_severity = "Warning"
                    else:
                        state.alert_severity = "Normal"

                state.last_detected_time = current_timestamp

        # Handle animals NOT detected in this frame (grace period / loss handling)
        missing_ids = set(self.animals.keys()) - seen_ids
        for m_id in missing_ids:
            state = self.animals[m_id]
            time_since_last_seen = current_timestamp - state.last_detected_time

            if time_since_last_seen > self.grace_period_seconds:
                state.status = "Lost"
                # Do NOT accumulate missing time as lying time!
            else:
                state.status = "GracePeriod"
                # Grace period prevents fluttering, but does not confirm posture

        return list(self.animals.values()), completed_events
