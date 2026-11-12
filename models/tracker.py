"""
Individual Animal Tracking Module Using ByteTrack
AI-Based Real-Time Livestock Behavior & Inactivity Monitoring System
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from config import COCO_COW_CLASS_ID, COCO_COW_CLASS_NAME
from models.cow_detector import CowDetection

logger = logging.getLogger(__name__)


@dataclass
class TrackedCow:
    """Representation of an actively tracked cow in the current frame."""
    tracking_id: int
    bbox: Tuple[int, int, int, int]
    confidence: float
    frame_number: int
    timestamp_seconds: float
    behavior: str = "Unknown"
    behavior_confidence: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tracking_id": self.tracking_id,
            "bbox": list(self.bbox),
            "confidence": round(float(self.confidence), 4),
            "frame_number": self.frame_number,
            "timestamp_seconds": round(float(self.timestamp_seconds), 3),
            "behavior": self.behavior,
            "behavior_confidence": round(float(self.behavior_confidence), 4) if self.behavior_confidence is not None else None,
        }


class ByteTrackerManager:
    """
    Manages multi-object tracking of detected cows across frames using ByteTrack.
    Utilizes Ultralytics' integrated ByteTrack implementation with persistence.
    """

    def __init__(self, detector_model=None):
        self.model = detector_model
        self.active_tracks: Dict[int, TrackedCow] = {}
        self.total_tracked_count: int = 0
        self._known_ids: set = set()

    def set_model(self, model) -> None:
        """Sets or updates the underlying YOLOv8 model instance."""
        self.model = model

    def reset(self) -> None:
        """Resets tracker state between different video streams."""
        self.active_tracks.clear()
        self.total_tracked_count = 0
        self._known_ids.clear()

    def track_frame(
        self,
        frame: np.ndarray,
        frame_number: int,
        timestamp_seconds: float,
        conf_threshold: float = 0.40,
        iou_threshold: float = 0.45,
    ) -> List[TrackedCow]:
        """
        Runs YOLOv8 detection + ByteTrack tracking on the input frame.
        Maintains IDs across frames and extracts bounding boxes and tracking IDs.
        """
        if self.model is None or frame is None or frame.size == 0:
            return []

        try:
            # Ultralytics track with persist=True maintains ByteTrack trajectories across sequential frames
            results = self.model.track(
                source=frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=conf_threshold,
                iou=iou_threshold,
                classes=[COCO_COW_CLASS_ID],
                verbose=False,
            )

            tracked_cows: List[TrackedCow] = []
            if not results or len(results) == 0:
                self.active_tracks.clear()
                return tracked_cows

            boxes = results[0].boxes
            if boxes is None or len(boxes) == 0:
                self.active_tracks.clear()
                return tracked_cows

            current_ids = set()

            for box in boxes:
                # Check if a tracking ID was assigned by ByteTrack
                if box.id is None:
                    continue

                track_id = int(box.id[0].cpu().numpy())
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = [int(round(float(v))) for v in xyxy]
                conf = float(box.conf[0].cpu().numpy())

                current_ids.add(track_id)
                self._known_ids.add(track_id)

                cow = TrackedCow(
                    tracking_id=track_id,
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    frame_number=frame_number,
                    timestamp_seconds=timestamp_seconds,
                )
                tracked_cows.append(cow)
                self.active_tracks[track_id] = cow

            # Remove tracks that disappeared from the immediate active set
            disappeared = set(self.active_tracks.keys()) - current_ids
            for dis_id in disappeared:
                self.active_tracks.pop(dis_id, None)

            self.total_tracked_count = len(self._known_ids)
            return tracked_cows

        except Exception as e:
            logger.error("Error during ByteTrack tracking inference: %s", e)
            return []
