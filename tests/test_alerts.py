"""
Unit Tests for Alert Generation Engine
AI-Based Real-Time Livestock Behavior & Inactivity Monitoring System
"""

import pytest
from processing.inactivity_engine import AnimalActivityState
from processing.alert_engine import AlertEngine


def test_alert_generation_and_escalation():
    """Verifies alert triggering on Warning and Critical states."""
    engine = AlertEngine(cooldown_seconds=30.0)

    # State with Normal severity
    state_normal = AnimalActivityState(
        tracking_id=1,
        first_detected_time=0.0,
        last_detected_time=5.0,
        last_frame_number=100,
        current_behavior="Standing",
        alert_severity="Normal",
    )
    alerts, resolved = engine.process_states([state_normal], current_timestamp=5.0)
    assert len(alerts) == 0
    assert len(resolved) == 0

    # State escalates to Warning
    state_warning = AnimalActivityState(
        tracking_id=1,
        first_detected_time=0.0,
        last_detected_time=12.0,
        last_frame_number=240,
        current_behavior="Lying/Resting",
        current_lying_duration=12.0,
        alert_severity="Warning",
    )
    alerts, resolved = engine.process_states([state_warning], current_timestamp=12.0)
    assert len(alerts) == 1
    assert alerts[0].severity == "Warning"
    assert alerts[0].tracking_id == 1

    # Immediate next frame at 12.1s (cooldown active) should not duplicate
    alerts2, _ = engine.process_states([state_warning], current_timestamp=12.1)
    assert len(alerts2) == 0

    # Escalation to Critical bypasses cooldown
    state_critical = AnimalActivityState(
        tracking_id=1,
        first_detected_time=0.0,
        last_detected_time=16.0,
        last_frame_number=320,
        current_behavior="Lying/Resting",
        current_lying_duration=16.0,
        alert_severity="Critical",
    )
    alerts3, _ = engine.process_states([state_critical], current_timestamp=16.0)
    assert len(alerts3) == 1
    assert alerts3[0].severity == "Critical"


def test_automatic_alert_resolution():
    """Verifies that an open alert resolves when the animal returns to Normal state."""
    engine = AlertEngine(cooldown_seconds=30.0)

    state_crit = AnimalActivityState(
        tracking_id=5,
        first_detected_time=0.0,
        last_detected_time=20.0,
        last_frame_number=400,
        current_behavior="Lying/Resting",
        current_lying_duration=20.0,
        alert_severity="Critical",
    )
    alerts, _ = engine.process_states([state_crit], current_timestamp=20.0)
    assert len(alerts) == 1
    assert 5 in engine.active_alert_ids

    # Cow stands up at t = 22.0s
    state_stood = AnimalActivityState(
        tracking_id=5,
        first_detected_time=0.0,
        last_detected_time=22.0,
        last_frame_number=440,
        current_behavior="Standing",
        current_lying_duration=0.0,
        alert_severity="Normal",
    )
    new_alerts, resolved_ids = engine.process_states([state_stood], current_timestamp=22.0)
    assert len(new_alerts) == 0
    assert 5 in resolved_ids
    assert 5 not in engine.active_alert_ids
