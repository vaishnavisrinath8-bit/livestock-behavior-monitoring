"""
Integration and Validation Tests for Video Processing Pipeline
AI-Based Real-Time Livestock Behavior & Inactivity Monitoring System
"""

import pytest
from pathlib import Path
import numpy as np
import cv2

from utils.video_utils import validate_video_file, get_video_metadata
from utils.validators import validate_processing_parameters, validate_feeding_zone
from models.cow_detector import CowDetector
from models.tracker import ByteTrackerManager
from models.behavior_classifier import LivestockBehaviorClassifier
from processing.inactivity_engine import InactivityEngine
from processing.alert_engine import AlertEngine
from processing.frame_processor import FrameProcessor
from database.database_operations import DatabaseManager
from database.db import init_database


def test_invalid_and_empty_video_handling(tmp_path):
    """Verifies that non-existent and 0-byte video files are caught by validators."""
    # 1. Non-existent file
    missing = tmp_path / "ghost.mp4"
    valid, msg = validate_video_file(missing)
    assert not valid
    assert "does not exist" in msg

    # 2. 0-byte empty file
    empty_file = tmp_path / "empty.mp4"
    empty_file.touch()
    valid, msg = validate_video_file(empty_file)
    assert not valid
    assert "empty" in msg

    # 3. Invalid extension
    wrong_ext = tmp_path / "audio.mp3"
    wrong_ext.write_text("dummy")
    valid, msg = validate_video_file(wrong_ext)
    assert not valid
    assert "Unsupported format" in msg


def test_parameter_validators():
    """Verifies range validations for user settings."""
    valid, _ = validate_processing_parameters(confidence=0.5, inactivity_threshold=100.0, cooldown=30.0, frame_interval=1)
    assert valid

    # Negative threshold
    invalid, msg = validate_processing_parameters(confidence=0.5, inactivity_threshold=-5.0, cooldown=30.0, frame_interval=1)
    assert not invalid

    # Invalid feeding zone (xmin >= xmax)
    invalid_z, _ = validate_feeding_zone({"xmin": 0.8, "ymin": 0.1, "xmax": 0.2, "ymax": 0.9})
    assert not invalid_z


def test_full_synthetic_frame_processing_pipeline(tmp_path):
    """Executes the complete FrameProcessor on a synthetic frame and confirms state persistence."""
    test_db = tmp_path / "pipeline_test.db"
    init_database(test_db)
    db_mgr = DatabaseManager(test_db)

    detector = CowDetector()
    tracker = ByteTrackerManager(detector.model)
    classifier = LivestockBehaviorClassifier()
    inact_engine = InactivityEngine(inactivity_threshold_seconds=10.0)
    alert_engine = AlertEngine(cooldown_seconds=15.0)

    processor = FrameProcessor(
        detector=detector,
        tracker=tracker,
        classifier=classifier,
        inactivity_engine=inact_engine,
        alert_engine=alert_engine,
        db_manager=db_mgr,
    )

    # Synthetic frame (e.g. 720p image)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    res = processor.process_frame(frame, frame_number=1, timestamp_seconds=0.04)

    assert res.annotated_frame is not None
    assert res.frame_number == 1
    assert res.timestamp_seconds == 0.04
    assert isinstance(res.cows_count, int)
