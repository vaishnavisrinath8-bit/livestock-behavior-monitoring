"""
Unit Tests for ByteTrack Tracking Module
AI-Based Real-Time Livestock Behavior & Inactivity Monitoring System
"""

import pytest
import numpy as np
from models.tracker import ByteTrackerManager, TrackedCow


def test_tracked_cow_dataclass():
    """Verifies TrackedCow dataclass fields and dictionary representation."""
    cow = TrackedCow(
        tracking_id=1,
        bbox=(100, 150, 300, 400),
        confidence=0.92,
        frame_number=45,
        timestamp_seconds=1.8,
        behavior="Standing",
    )
    d = cow.to_dict()
    assert d["tracking_id"] == 1
    assert d["bbox"] == [100, 150, 300, 400]
    assert d["behavior"] == "Standing"
    assert d["confidence"] == 0.92


def test_tracker_reset_and_empty_handling():
    """Verifies that reset clears internal active track structures."""
    tracker = ByteTrackerManager(detector_model=None)
    assert tracker.total_tracked_count == 0

    # Running on None model returns empty gracefully
    res = tracker.track_frame(np.zeros((480, 640, 3), dtype=np.uint8), frame_number=1, timestamp_seconds=0.04)
    assert res == []

    tracker.reset()
    assert len(tracker.active_tracks) == 0
