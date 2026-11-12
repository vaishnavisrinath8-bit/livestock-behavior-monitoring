"""
Cow Detection Module Using YOLOv8
AI-Based Real-Time Livestock Behavior & Inactivity Monitoring System
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from config import (
    DEFAULT_DETECTION_MODEL,
    COCO_COW_CLASS_ID,
    COCO_COW_CLASS_NAME,
    WEIGHTS_DIR,
)

logger = logging.getLogger(__name__)


@dataclass
class CowDetection:
    """Standardized representation of a single detected cow."""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) in pixel coordinates
    confidence: float
    class_id: int
    class_name: str
    frame_number: int
    timestamp_seconds: float
    tracking_id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bbox": list(self.bbox),
            "confidence": round(float(self.confidence), 4),
            "class_id": self.class_id,
            "class_name": self.class_name,
            "frame_number": self.frame_number,
            "timestamp_seconds": round(float(self.timestamp_seconds), 3),
            "tracking_id": self.tracking_id,
        }


class CowDetector:
    """
    Wraps Ultralytics YOLOv8 for filtered cow detection.
    Filters specifically for COCO Class 19 ('cow') to ignore people,
    vehicles, machinery, or other non-livestock objects.
    """

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_DETECTION_MODEL,
        confidence_threshold: float = 0.40,
        iou_threshold: float = 0.45,
    ):
        self.model_name_or_path = model_name_or_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """Loads YOLOv8 model instance with safety and logging."""
        try:
            from ultralytics import YOLO

            # Check if weights file exists locally in weights dir or root, else let YOLO download
            weights_local = WEIGHTS_DIR / self.model_name_or_path
            path_to_use = str(weights_local) if weights_local.exists() else self.model_name_or_path

            logger.info("Loading YOLOv8 cow detector from %s", path_to_use)
            self.model = YOLO(path_to_use)
            logger.info("YOLOv8 cow detector loaded successfully.")
        except Exception as e:
            logger.error("Failed to load YOLO model '%s': %s", self.model_name_or_path, e)
            self.model = None

    def is_ready(self) -> bool:
        """Returns True if model weights are loaded and ready for inference."""
        return self.model is not None

    def detect(
        self,
        frame: np.ndarray,
        frame_number: int = 0,
        timestamp_seconds: float = 0.0,
        confidence_threshold: Optional[float] = None,
    ) -> List[CowDetection]:
        """
        Runs object detection on a single frame and filters for 'cow' class.
        Returns a list of CowDetection objects.
        """
        if not self.is_ready():
            logger.warning("CowDetector invoked before model is loaded.")
            return []

        if frame is None or frame.size == 0:
            return []

        conf = confidence_threshold if confidence_threshold is not None else self.confidence_threshold

        try:
            # classes=[19] instructs YOLO to predict and filter for cow only at NMS level
            results = self.model.predict(
                source=frame,
                conf=conf,
                iou=self.iou_threshold,
                classes=[COCO_COW_CLASS_ID],
                verbose=False,
            )

            detections: List[CowDetection] = []
            if not results or len(results) == 0:
                return detections

            result = results[0]
            boxes = result.boxes

            if boxes is None or len(boxes) == 0:
                return detections

            for box in boxes:
                # Extract coordinates
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = [int(round(float(v))) for v in xyxy]
                score = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())

                if cls_id == COCO_COW_CLASS_ID:
                    detections.append(
                        CowDetection(
                            bbox=(x1, y1, x2, y2),
                            confidence=score,
                            class_id=cls_id,
                            class_name=COCO_COW_CLASS_NAME,
                            frame_number=frame_number,
                            timestamp_seconds=timestamp_seconds,
                        )
                    )

            return detections
        except Exception as e:
            logger.error("Error during cow detection inference: %s", e)
            return []
