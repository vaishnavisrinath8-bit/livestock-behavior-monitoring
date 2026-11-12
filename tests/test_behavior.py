"""
Unit Tests for Behavior Classifier
AI-Based Real-Time Livestock Behavior & Inactivity Monitoring System
"""

import numpy as np
import pytest
from pathlib import Path
from models.behavior_classifier import LivestockBehaviorClassifier
from config import (
    BEHAVIOR_STANDING,
    BEHAVIOR_FEEDING,
    BEHAVIOR_LYING,
    BEHAVIOR_UNKNOWN,
)


def test_missing_custom_model_fallback():
    """Verifies that missing custom weights smoothly activates Baseline Estimation."""
    non_existent = Path("non_existent_weights.pt")
    classifier = LivestockBehaviorClassifier(model_path=non_existent)

    assert classifier.mode == "BASELINE_ESTIMATION"
    assert classifier.custom_model is None


def test_baseline_feeding_zone_detection():
    """Verifies that bounding boxes located in the feeding bunk zone are classified as Feeding."""
    classifier = LivestockBehaviorClassifier()
    synthetic_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    # Feeding zone located at bottom half of screen
    zone = {"xmin": 0.0, "ymin": 0.5, "xmax": 1.0, "ymax": 1.0}

    # Cow whose bottom is in the feeding zone (y2 = 600 > 0.5 * 720 = 360)
    bbox = (200, 300, 400, 600)
    pred = classifier.classify_crop(synthetic_frame, bbox, feeding_zone=zone)

    assert pred.behavior == BEHAVIOR_FEEDING
    assert pred.confidence is None  # Never fabricated for heuristics
    assert pred.mode == "BASELINE_ESTIMATION"


def test_baseline_posture_heuristics():
    """Verifies geometric posture estimation for pronounced lying vs standing aspect ratios."""
    classifier = LivestockBehaviorClassifier()
    synthetic_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    # 1. Wide and low profile (e.g. width=400, height=180 -> aspect ratio ~ 2.22) -> Lying
    bbox_lying = (100, 100, 500, 280)
    pred_lying = classifier.classify_crop(synthetic_frame, bbox_lying, feeding_zone=None)
    assert pred_lying.behavior == BEHAVIOR_LYING

    # 2. Vertical profile (e.g. width=200, height=250 -> aspect ratio = 0.8) -> Standing
    bbox_standing = (100, 100, 300, 350)
    pred_standing = classifier.classify_crop(synthetic_frame, bbox_standing, feeding_zone=None)
    assert pred_standing.behavior == BEHAVIOR_STANDING

    # 3. Ambiguous posture (e.g. width=300, height=200 -> aspect ratio = 1.5) -> Unknown
    bbox_ambiguous = (100, 100, 400, 300)
    pred_ambiguous = classifier.classify_crop(synthetic_frame, bbox_ambiguous, feeding_zone=None)
    assert pred_ambiguous.behavior == BEHAVIOR_UNKNOWN
