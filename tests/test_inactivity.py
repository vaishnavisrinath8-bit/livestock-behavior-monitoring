"""
Unit Tests for Inactivity Engine
AI-Based Real-Time Livestock Behavior & Inactivity Monitoring System
"""

import pytest
from processing.inactivity_engine import InactivityEngine
from config import (
    BEHAVIOR_STANDING,
    BEHAVIOR_FEEDING,
    BEHAVIOR_LYING,
    BEHAVIOR_UNKNOWN,
)
from utils.time_utils import calculate_video_timestamp


def test_continuous_lying_duration_accumulation():
    """Verifies that continuous lying timer accumulates strictly during Lying/Resting."""
    engine = InactivityEngine(inactivity_threshold_seconds=10.0, grace_period_seconds=2.0)

    # Frame 1: Cow 1 detected Lying at t = 0.0s
    engine.update([{"tracking_id": 1, "behavior": BEHAVIOR_LYING}], current_timestamp=0.0, frame_number=0)
    assert engine.animals[1].current_lying_duration == 0.0

    # Frame 2: Cow 1 continues Lying at t = 5.0s (5.0s < 7.0s warning threshold)
    engine.update([{"tracking_id": 1, "behavior": BEHAVIOR_LYING}], current_timestamp=5.0, frame_number=100)
    assert engine.animals[1].current_lying_duration == 5.0
    assert engine.animals[1].alert_severity == "Normal"

    # Frame 3: Cow 1 continues Lying at t = 7.5s (7.5s >= 7.0s warning threshold)
    engine.update([{"tracking_id": 1, "behavior": BEHAVIOR_LYING}], current_timestamp=7.5, frame_number=150)
    assert engine.animals[1].current_lying_duration == 7.5
    assert engine.animals[1].alert_severity == "Warning"

    # Frame 3: Cow 1 crosses threshold at t = 11.0s
    engine.update([{"tracking_id": 1, "behavior": BEHAVIOR_LYING}], current_timestamp=11.0, frame_number=220)
    assert engine.animals[1].current_lying_duration == 11.0
    assert engine.animals[1].alert_severity == "Critical"


def test_timer_reset_on_behavior_transition():
    """Verifies that continuous lying timer resets immediately when animal stands up."""
    engine = InactivityEngine(inactivity_threshold_seconds=10.0)

    # Accumulate lying duration
    engine.update([{"tracking_id": 2, "behavior": BEHAVIOR_LYING}], current_timestamp=0.0, frame_number=0)
    engine.update([{"tracking_id": 2, "behavior": BEHAVIOR_LYING}], current_timestamp=12.0, frame_number=240)
    assert engine.animals[2].current_lying_duration == 12.0
    assert engine.animals[2].alert_severity == "Critical"

    # Animal stands up at t = 13.0s
    states, completed_events = engine.update(
        [{"tracking_id": 2, "behavior": BEHAVIOR_STANDING}],
        current_timestamp=13.0,
        frame_number=260,
    )

    assert engine.animals[2].current_lying_duration == 0.0
    assert engine.animals[2].alert_severity == "Normal"
    assert engine.animals[2].current_behavior == BEHAVIOR_STANDING
    assert len(completed_events) == 1
    assert completed_events[0]["behavior"] == BEHAVIOR_LYING
    assert completed_events[0]["duration_seconds"] == 13.0


def test_tracking_loss_grace_period():
    """Verifies that missing detections do not falsely inflate lying duration."""
    engine = InactivityEngine(inactivity_threshold_seconds=15.0, grace_period_seconds=2.0)

    # Cow 3 detected lying at t = 0.0s
    engine.update([{"tracking_id": 3, "behavior": BEHAVIOR_LYING}], current_timestamp=0.0, frame_number=0)
    engine.update([{"tracking_id": 3, "behavior": BEHAVIOR_LYING}], current_timestamp=4.0, frame_number=80)
    assert engine.animals[3].current_lying_duration == 4.0

    # Cow 3 is occluded / disappears between t = 4.0s and t = 10.0s (6 seconds > grace period)
    engine.update([], current_timestamp=10.0, frame_number=200)

    # Animal must be marked Lost, and lying duration must NOT have accumulated the 6 missing seconds
    assert engine.animals[3].status == "Lost"
    assert engine.animals[3].current_lying_duration == 4.0


def test_video_timestamp_calculation():
    """Verifies precise video timestamp formula frame / fps."""
    ts1 = calculate_video_timestamp(frame_number=50, fps=25.0)
    assert ts1 == 2.0

    ts2 = calculate_video_timestamp(frame_number=120, fps=30.0)
    assert ts2 == 4.0
