"""
Unit and Integration Tests for Cow Detector Module
AI-Based Real-Time Livestock Behavior & Inactivity Monitoring System
"""

import pytest
import numpy as np
from models.cow_detector import CowDetector, CowDetection
from config import COCO_COW_CLASS_ID, COCO_COW_CLASS_NAME


def test_cow_detection_dataclass():
    """Verifies CowDetection object structure and dictionary serialization."""
    det = CowDetection(
        bbox=(50, 60, 200, 250),
        confidence=0.8876,
        class_id=COCO_COW_CLASS_ID,
        class_name=COCO_COW_CLASS_NAME,
        frame_number=12,
        timestamp_seconds=0.48,
        tracking_id=3,
    )
    d = det.to_dict()
    assert d["bbox"] == [50, 60, 200, 250]
    assert d["confidence"] == 0.8876
    assert d["class_name"] == "cow"
    assert d["tracking_id"] == 3


def test_detector_initialization_and_empty_frame():
    """Verifies that passing an empty frame to CowDetector returns an empty list without error."""
    detector = CowDetector()
    # Empty or None frame handling
    res_none = detector.detect(None)
    assert res_none == []

    res_empty = detector.detect(np.zeros((0, 0, 3), dtype=np.uint8))
    assert res_empty == []
