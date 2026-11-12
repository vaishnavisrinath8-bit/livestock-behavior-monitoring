"""
Time and Timestamp Calculation Utilities
AI-Based Real-Time Livestock Behavior & Inactivity Monitoring System
"""

from datetime import datetime, timezone
from typing import Tuple


def get_iso_timestamp() -> str:
    """Returns current UTC ISO-formatted timestamp."""
    return datetime.now(timezone.utc).isoformat()


def seconds_to_hms(seconds: float) -> str:
    """
    Formats a duration in seconds into human-readable HH:MM:SS or MM:SS format.
    """
    if seconds is None or seconds < 0:
        return "00:00"
    total_sec = int(round(seconds))
    hours = total_sec // 3600
    minutes = (total_sec % 3600) // 60
    sec = total_sec % 60

    if hours > 0:
        return f"{hours:02d}h {minutes:02d}m {sec:02d}s"
    return f"{minutes:02d}m {sec:02d}s"


def format_duration_short(seconds: float) -> str:
    """Short format: e.g. 15.2s, 4m 12s, 1h 20m."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    return seconds_to_hms(seconds)


def calculate_video_timestamp(frame_number: int, fps: float) -> float:
    """
    Computes exact video playback timestamp in seconds from frame index and FPS.
    Prevents tying inactivity duration to laptop processing speed.
    """
    if fps <= 0:
        return 0.0
    return float(frame_number) / fps
