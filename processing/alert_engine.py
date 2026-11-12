"""
Alert Engine Module
AI-Based Real-Time Livestock Behavior & Inactivity Monitoring System
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from config import (
    DEFAULT_ALERT_COOLDOWN_SECONDS,
    BEHAVIOR_LYING,
)
from utils.time_utils import seconds_to_hms, get_iso_timestamp
from processing.inactivity_engine import AnimalActivityState

logger = logging.getLogger(__name__)


@dataclass
class LivestockAlert:
    """Represents a generated observation alert event."""
    tracking_id: int
    alert_type: str
    inactivity_seconds: float
    severity: str  # "Warning" or "Critical"
    message: str
    created_at: str
    resolved: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tracking_id": self.tracking_id,
            "alert_type": self.alert_type,
            "inactivity_seconds": round(self.inactivity_seconds, 2),
            "severity": self.severity,
            "message": self.message,
            "created_at": self.created_at,
            "resolved": self.resolved,
        }


class AlertEngine:
    """
    Manages alert state transitions, rate limits alerts via configurable cooldowns,
    prevents duplicate per-frame spam, and triggers automatic resolution.
    """

    def __init__(self, cooldown_seconds: float = DEFAULT_ALERT_COOLDOWN_SECONDS):
        self.cooldown_seconds = cooldown_seconds
        # tracking_id -> last_alert_time (video seconds or wall-clock)
        self.last_alert_timestamps: Dict[int, float] = {}
        # tracking_id -> last alerted severity
        self.last_alert_severity: Dict[int, str] = {}
        # Set of tracking_ids with active unresolved alerts
        self.active_alert_ids: set = set()

    def set_cooldown(self, seconds: float) -> None:
        """Sets alert cooldown period in seconds."""
        if seconds >= 0:
            self.cooldown_seconds = float(seconds)

    def reset(self) -> None:
        """Resets engine state."""
        self.last_alert_timestamps.clear()
        self.last_alert_severity.clear()
        self.active_alert_ids.clear()

    def process_states(
        self,
        states: List[AnimalActivityState],
        current_timestamp: float,
    ) -> Tuple[List[LivestockAlert], List[int]]:
        """
        Evaluates current animal states for alerts and automatic resolutions.
        Returns:
            new_alerts: list of newly triggered LivestockAlert objects
            resolved_ids: list of tracking_ids whose prolonged inactivity has ended
        """
        new_alerts: List[LivestockAlert] = []
        resolved_ids: List[int] = []

        for state in states:
            track_id = state.tracking_id
            severity = state.alert_severity

            # Case 1: Animal has returned to Normal (e.g. stood up or feeding)
            if severity == "Normal":
                if track_id in self.active_alert_ids:
                    # Condition ended - trigger resolution
                    self.active_alert_ids.remove(track_id)
                    self.last_alert_severity.pop(track_id, None)
                    resolved_ids.append(track_id)
                    logger.info("Alert automatically resolved for Cow #%d (returned to %s)", track_id, state.current_behavior)
                continue

            # Case 2: Animal is in Warning or Critical inactivity
            if severity in ("Warning", "Critical"):
                last_time = self.last_alert_timestamps.get(track_id, -9999.0)
                last_sev = self.last_alert_severity.get(track_id, "Normal")
                time_since_alert = current_timestamp - last_time

                # Trigger condition:
                # A) Escalated severity (e.g. from Warning to Critical) OR
                # B) Cooldown period has elapsed and condition persists
                should_trigger = False
                if severity == "Critical" and last_sev != "Critical":
                    should_trigger = True
                elif time_since_alert >= self.cooldown_seconds and (track_id not in self.active_alert_ids or severity != last_sev):
                    should_trigger = True

                if should_trigger:
                    dur_str = seconds_to_hms(state.current_lying_duration)
                    if severity == "Critical":
                        msg = f"Cow #{track_id} has remained in Lying/Resting state for {dur_str} (exceeded threshold)."
                    else:
                        msg = f"Cow #{track_id} prolonged lying observed ({dur_str}). Approaching threshold."

                    alert = LivestockAlert(
                        tracking_id=track_id,
                        alert_type="Prolonged Inactivity",
                        inactivity_seconds=state.current_lying_duration,
                        severity=severity,
                        message=msg,
                        created_at=get_iso_timestamp(),
                    )
                    new_alerts.append(alert)

                    self.last_alert_timestamps[track_id] = current_timestamp
                    self.last_alert_severity[track_id] = severity
                    self.active_alert_ids.add(track_id)

        return new_alerts, resolved_ids
