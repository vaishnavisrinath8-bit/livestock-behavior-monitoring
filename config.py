"""
System Configuration Module
AI-Based Real-Time Livestock Behavior & Inactivity Monitoring System
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# Base Project Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLE_VIDEOS_DIR = DATA_DIR / "sample_videos"
WEIGHTS_DIR = BASE_DIR / "weights"
ASSETS_DIR = BASE_DIR / "assets"
DB_PATH = DATA_DIR / "livestock.db"

# Ensure runtime directories exist
for directory in [DATA_DIR, UPLOADS_DIR, PROCESSED_DIR, SAMPLE_VIDEOS_DIR, WEIGHTS_DIR, ASSETS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Detection Model Settings
DEFAULT_DETECTION_MODEL = "yolov8n.pt"
COCO_COW_CLASS_ID = 19  # COCO dataset index for 'cow'
COCO_COW_CLASS_NAME = "cow"
DEFAULT_CONFIDENCE_THRESHOLD = 0.40
DEFAULT_IOU_THRESHOLD = 0.45

# Custom Behavior Model Settings
CUSTOM_BEHAVIOR_MODEL_PATH = WEIGHTS_DIR / "behavior_model.pt"

# Recognized Behavior Classes
BEHAVIOR_STANDING = "Standing"
BEHAVIOR_FEEDING = "Feeding"
BEHAVIOR_LYING = "Lying/Resting"
BEHAVIOR_UNKNOWN = "Unknown"

VALID_BEHAVIORS: List[str] = [
    BEHAVIOR_STANDING,
    BEHAVIOR_FEEDING,
    BEHAVIOR_LYING,
    BEHAVIOR_UNKNOWN,
]

# Color Palette (BGR for OpenCV, Hex for Web/Plotly)
BEHAVIOR_COLORS_BGR: Dict[str, Tuple[int, int, int]] = {
    BEHAVIOR_STANDING: (40, 180, 40),      # Green
    BEHAVIOR_FEEDING: (220, 120, 20),      # Blue
    BEHAVIOR_LYING: (30, 215, 255),       # Yellow
    BEHAVIOR_UNKNOWN: (140, 140, 140),     # Gray
    "Alert_Critical": (30, 30, 220),      # Red
    "Alert_Warning": (0, 140, 255),       # Orange
}

BEHAVIOR_COLORS_HEX: Dict[str, str] = {
    BEHAVIOR_STANDING: "#28a745",
    BEHAVIOR_FEEDING: "#17a2b8",
    BEHAVIOR_LYING: "#ffc107",
    BEHAVIOR_UNKNOWN: "#6c757d",
    "Alert_Critical": "#dc3545",
    "Alert_Warning": "#fd7e14",
}

# Inactivity & Alert Defaults
DEFAULT_INACTIVITY_THRESHOLD_SECONDS = 900.0  # 15 minutes
DEMO_INACTIVITY_THRESHOLD_SECONDS = 10.0      # Demonstration mode: 10 seconds
WARNING_THRESHOLD_RATIO = 0.70                # Warning triggered at 70% of threshold
DEFAULT_ALERT_COOLDOWN_SECONDS = 30.0         # Seconds before re-alerting for the same condition
DEFAULT_TRACKING_GRACE_PERIOD_SECONDS = 3.0   # Grace period for temporary occlusion/loss

# Supported Video Formats
SUPPORTED_VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov"]

# UI & Dashboard Styling
DASHBOARD_THEME = {
    "primaryColor": "#1b5e20",       # Dark Green
    "backgroundColor": "#f8f9fa",    # Off-white / light neutral
    "secondaryBackgroundColor": "#ffffff",
    "textColor": "#212529",
    "font": "sans-serif",
}


@dataclass
class ProcessingConfig:
    """Configurable runtime parameters for the video processing pipeline."""
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    iou_threshold: float = DEFAULT_IOU_THRESHOLD
    inactivity_threshold_seconds: float = DEFAULT_INACTIVITY_THRESHOLD_SECONDS
    alert_cooldown_seconds: float = DEFAULT_ALERT_COOLDOWN_SECONDS
    tracking_grace_period_seconds: float = DEFAULT_TRACKING_GRACE_PERIOD_SECONDS
    frame_interval: int = 1                    # Process every Nth frame (1 = every frame)
    target_width: int = 960                    # Optimal frame width for balanced laptop performance
    feeding_zone: Dict[str, float] = field(default_factory=dict)  # Optional normalized bbox {ymin, ymax, xmin, xmax}
    custom_model_path: str = str(CUSTOM_BEHAVIOR_MODEL_PATH)
    demo_mode: bool = False

    def get_effective_inactivity_threshold(self) -> float:
        return DEMO_INACTIVITY_THRESHOLD_SECONDS if self.demo_mode else self.inactivity_threshold_seconds
