"""
Single Frame Processing Pipeline
AI-Based Real-Time Livestock Behavior & Inactivity Monitoring System
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import cv2

from config import (
    ProcessingConfig,
    BEHAVIOR_COLORS_BGR,
    BEHAVIOR_STANDING,
    BEHAVIOR_FEEDING,
    BEHAVIOR_LYING,
    BEHAVIOR_UNKNOWN,
)
from models.cow_detector import CowDetector
from models.tracker import ByteTrackerManager, TrackedCow
from models.behavior_classifier import LivestockBehaviorClassifier
from processing.inactivity_engine import InactivityEngine, AnimalActivityState
from processing.alert_engine import AlertEngine, LivestockAlert
from database.database_operations import DatabaseManager
from utils.image_utils import resize_frame, draw_bounding_box
from utils.time_utils import get_iso_timestamp

logger = logging.getLogger(__name__)


@dataclass
class ProcessedFrameResult:
    """Carries visual and structured analytical results for a single processed frame."""
    annotated_frame: np.ndarray
    tracked_cows: List[TrackedCow]
    active_states: List[AnimalActivityState]
    new_alerts: List[LivestockAlert]
    frame_number: int
    timestamp_seconds: float
    cows_count: int


class FrameProcessor:
    """
    Executes the 12-step sequential analysis on each video or webcam frame.
    Coordinates detection, tracking, behavior classification, inactivity counting,
    alert thresholds, and database synchronization.
    """

    def __init__(
        self,
        detector: CowDetector,
        tracker: ByteTrackerManager,
        classifier: LivestockBehaviorClassifier,
        inactivity_engine: InactivityEngine,
        alert_engine: AlertEngine,
        db_manager: Optional[DatabaseManager] = None,
        config: Optional[ProcessingConfig] = None,
    ):
        self.detector = detector
        self.tracker = tracker
        self.classifier = classifier
        self.inactivity_engine = inactivity_engine
        self.alert_engine = alert_engine
        self.db_manager = db_manager
        self.config = config or ProcessingConfig()

        # Connect detector model to tracker
        if self.detector.is_ready():
            self.tracker.set_model(self.detector.model)

    def process_frame(
        self,
        frame: np.ndarray,
        frame_number: int,
        timestamp_seconds: float,
        source_type: str = "video",
    ) -> ProcessedFrameResult:
        """Processes a single raw frame through the complete AI monitoring pipeline."""
        if frame is None or frame.size == 0:
            raise ValueError("Input frame is invalid or empty.")

        # 1. & 2. Preprocess frame (proportional resize for laptop efficiency)
        processed_frame = resize_frame(frame, target_width=self.config.target_width)
        annotated_frame = processed_frame.copy()

        # Update inactivity threshold in engine based on config / demo mode
        effective_threshold = self.config.get_effective_inactivity_threshold()
        self.inactivity_engine.set_inactivity_threshold(effective_threshold)
        self.alert_engine.set_cooldown(self.config.alert_cooldown_seconds)

        # Ensure tracker has the detector model
        if self.tracker.model is None and self.detector.is_ready():
            self.tracker.set_model(self.detector.model)

        # 3. & 4. Track cows using ByteTrack
        tracked_cows = self.tracker.track_frame(
            frame=processed_frame,
            frame_number=frame_number,
            timestamp_seconds=timestamp_seconds,
            conf_threshold=self.config.confidence_threshold,
            iou_threshold=self.config.iou_threshold,
        )

        active_detections = []

        # 5. Classify observable behavior for each cow
        for cow in tracked_cows:
            pred = self.classifier.classify_crop(
                frame=processed_frame,
                bbox=cow.bbox,
                feeding_zone=self.config.feeding_zone,
            )
            cow.behavior = pred.behavior
            cow.behavior_confidence = pred.confidence  # None in baseline mode to prevent fake confidence

            active_detections.append({
                "tracking_id": cow.tracking_id,
                "behavior": cow.behavior,
            })

        # 6. & 7. Update activity records and continuous lying timers
        states, completed_events = self.inactivity_engine.update(
            active_detections=active_detections,
            current_timestamp=timestamp_seconds,
            frame_number=frame_number,
        )
        state_map = {s.tracking_id: s for s in states}

        # 8. & 9. Check alert thresholds and generate alerts
        new_alerts, resolved_ids = self.alert_engine.process_states(
            states=states,
            current_timestamp=timestamp_seconds,
        )

        # 10. Save relevant database records (SQLite)
        if self.db_manager is not None:
            iso_now = get_iso_timestamp()
            # Upsert active animals
            for cow in tracked_cows:
                st = state_map.get(cow.tracking_id)
                status_str = st.status if st else "Active"
                total_obs = st.total_observed_duration if st else 0.0
                self.db_manager.upsert_animal(
                    tracking_id=cow.tracking_id,
                    timestamp=iso_now,
                    current_behavior=cow.behavior,
                    total_observed_seconds=round(total_obs, 2),
                    status=status_str,
                )

            # Persist completed behavior logs
            for evt in completed_events:
                self.db_manager.insert_activity_log(
                    tracking_id=evt["tracking_id"],
                    behavior=evt["behavior"],
                    duration_seconds=evt["duration_seconds"],
                    timestamp=iso_now,
                    source_type=source_type,
                )

            # Persist new alerts
            for alert in new_alerts:
                self.db_manager.insert_alert(
                    tracking_id=alert.tracking_id,
                    alert_type=alert.alert_type,
                    inactivity_seconds=alert.inactivity_seconds,
                    severity=alert.severity,
                    message=alert.message,
                    created_at=alert.created_at,
                )

            # Resolve alerts
            for r_id in resolved_ids:
                self.db_manager.resolve_alerts_for_animal(r_id)

        # 11. Draw bounding boxes, labels, and status badges
        for cow in tracked_cows:
            st = state_map.get(cow.tracking_id)
            severity = st.alert_severity if st else "Normal"

            # Determine color
            if severity == "Critical":
                box_color = BEHAVIOR_COLORS_BGR["Alert_Critical"]
                banner_text = "WARNING: PROLONGED INACTIVITY"
            elif severity == "Warning":
                box_color = BEHAVIOR_COLORS_BGR["Alert_Warning"]
                banner_text = "CAUTION: PROLONGED LYING"
            else:
                box_color = BEHAVIOR_COLORS_BGR.get(cow.behavior, BEHAVIOR_COLORS_BGR[BEHAVIOR_UNKNOWN])
                banner_text = None

            # Only display behavior confidence if predicted by custom AI model
            conf_display = cow.behavior_confidence if (self.classifier.mode == "CUSTOM_AI") else None

            draw_bounding_box(
                frame=annotated_frame,
                bbox=cow.bbox,
                tracking_id=cow.tracking_id,
                behavior=cow.behavior,
                color=box_color,
                confidence=conf_display,
                alert_banner=banner_text,
            )

        return ProcessedFrameResult(
            annotated_frame=annotated_frame,
            tracked_cows=tracked_cows,
            active_states=states,
            new_alerts=new_alerts,
            frame_number=frame_number,
            timestamp_seconds=timestamp_seconds,
            cows_count=len(tracked_cows),
        )
