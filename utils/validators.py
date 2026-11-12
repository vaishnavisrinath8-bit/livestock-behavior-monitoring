"""
Input and Parameter Validation Utilities
AI-Based Real-Time Livestock Behavior & Inactivity Monitoring System
"""

from typing import Tuple, Dict, Any


def validate_processing_parameters(
    confidence: float,
    inactivity_threshold: float,
    cooldown: float,
    frame_interval: int,
) -> Tuple[bool, str]:
    """Validates user-adjustable thresholds from settings dashboard."""
    if not (0.05 <= confidence <= 0.95):
        return False, "Confidence threshold must be between 0.05 and 0.95."

    if inactivity_threshold <= 0.0:
        return False, "Inactivity threshold must be greater than 0 seconds."

    if cooldown < 0.0:
        return False, "Alert cooldown must be non-negative."

    if frame_interval < 1 or frame_interval > 30:
        return False, "Frame interval must be between 1 and 30."

    return True, "Valid configuration parameters"


def validate_feeding_zone(zone: Dict[str, float]) -> Tuple[bool, str]:
    """Validates normalized feeding zone coordinates."""
    if not zone:
        return True, "Empty zone is disabled"

    required_keys = {"xmin", "ymin", "xmax", "ymax"}
    if not required_keys.issubset(zone.keys()):
        return False, f"Feeding zone missing coordinates. Required: {required_keys}"

    for k in required_keys:
        val = zone[k]
        if not (0.0 <= val <= 1.0):
            return False, f"Coordinate {k} ({val}) must be in range [0.0, 1.0]."

    if zone["xmin"] >= zone["xmax"]:
        return False, "xmin must be less than xmax."

    if zone["ymin"] >= zone["ymax"]:
        return False, "ymin must be less than ymax."

    return True, "Valid feeding zone"
