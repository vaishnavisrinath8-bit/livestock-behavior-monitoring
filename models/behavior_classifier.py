"""
Livestock Behavior Classification Module
AI-Based Real-Time Livestock Behavior & Inactivity Monitoring System
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import numpy as np
import cv2

from config import (
    CUSTOM_BEHAVIOR_MODEL_PATH,
    BEHAVIOR_STANDING,
    BEHAVIOR_FEEDING,
    BEHAVIOR_LYING,
    BEHAVIOR_UNKNOWN,
)
from utils.image_utils import is_in_roi

logger = logging.getLogger(__name__)


@dataclass
class BehaviorPrediction:
    """Encapsulates behavior prediction outcome and transparency metadata."""
    behavior: str
    confidence: Optional[float]
    mode: str  # "CUSTOM_AI" or "BASELINE_ESTIMATION"
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "behavior": self.behavior,
            "confidence": round(float(self.confidence), 4) if self.confidence is not None else None,
            "mode": self.mode,
            "rationale": self.rationale,
        }


class LivestockBehaviorClassifier:
    """
    Modular Behavior Classifier supporting:
    Mode A: Custom-trained AI Behavior Model (weights/behavior_model.pt)
    Mode B: Explainable Baseline Estimation (when no custom weights are present)
    """

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = Path(model_path) if model_path else CUSTOM_BEHAVIOR_MODEL_PATH
        self.custom_model = None
        self.mode = "BASELINE_ESTIMATION"
        self._load_custom_model_if_available()

    def _load_custom_model_if_available(self) -> None:
        """Attempts to load custom trained weights if present on disk."""
        if self.model_path.exists() and self.model_path.stat().st_size > 0:
            try:
                from ultralytics import YOLO

                logger.info("Attempting to load custom behavior model from %s", self.model_path)
                model = YOLO(str(self.model_path))
                self.custom_model = model
                self.mode = "CUSTOM_AI"
                logger.info("Custom behavior model loaded successfully.")
            except Exception as e:
                logger.warning(
                    "Custom behavior model at %s could not be loaded: %s. Falling back to Baseline Estimation.",
                    self.model_path,
                    e,
                )
                self.custom_model = None
                self.mode = "BASELINE_ESTIMATION"
        else:
            self.custom_model = None
            self.mode = "BASELINE_ESTIMATION"
            logger.info("No custom behavior model found at %s. Operating in Baseline Estimation mode.", self.model_path)

    def classify_crop(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        feeding_zone: Optional[Dict[str, float]] = None,
    ) -> BehaviorPrediction:
        """
        Classifies animal behavior using Custom AI model if available,
        or rigorous, transparent baseline estimation if not.
        """
        # Mode A: Custom AI Model
        if self.mode == "CUSTOM_AI" and self.custom_model is not None:
            return self._predict_with_custom_model(frame, bbox)

        # Mode B: Baseline Estimation
        return self._predict_with_baseline(frame, bbox, feeding_zone)

    def _predict_with_custom_model(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> BehaviorPrediction:
        """Runs inference using a fine-tuned custom behavior model."""
        try:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                return BehaviorPrediction(
                    behavior=BEHAVIOR_UNKNOWN,
                    confidence=None,
                    mode="CUSTOM_AI",
                    rationale="Bounding box crop too small for feature extraction.",
                )

            # Classify cropped animal region
            results = self.custom_model(crop, verbose=False)
            if not results or len(results) == 0:
                return BehaviorPrediction(
                    behavior=BEHAVIOR_UNKNOWN,
                    confidence=None,
                    mode="CUSTOM_AI",
                    rationale="No inference output from custom model.",
                )

            res = results[0]
            # Handle classification output
            if hasattr(res, "probs") and res.probs is not None:
                top1_idx = int(res.probs.top1)
                conf = float(res.probs.top1conf.cpu().numpy())
                class_names = res.names
                predicted_name = class_names.get(top1_idx, BEHAVIOR_UNKNOWN)

                # Standardize to project categories
                norm_name = self._normalize_class_name(predicted_name)
                return BehaviorPrediction(
                    behavior=norm_name,
                    confidence=conf,
                    mode="CUSTOM_AI",
                    rationale=f"Classified by custom model with {conf*100:.1f}% confidence.",
                )

            return BehaviorPrediction(
                behavior=BEHAVIOR_UNKNOWN,
                confidence=None,
                mode="CUSTOM_AI",
                rationale="Model output format incompatible.",
            )

        except Exception as e:
            logger.error("Error running custom behavior model: %s", e)
            return BehaviorPrediction(
                behavior=BEHAVIOR_UNKNOWN,
                confidence=None,
                mode="CUSTOM_AI",
                rationale=f"Custom model error: {e}",
            )

    def _predict_with_baseline(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        feeding_zone: Optional[Dict[str, float]],
    ) -> BehaviorPrediction:
        """
        Baseline heuristic estimation:
        1. Checks feeding zone inclusion (feeding bunk).
        2. Evaluates aspect ratio (width vs. height) for pronounced lying postures.
        3. Assigns 'Unknown' when posture is ambiguous (never invents fake confidence).
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        box_w = max(1, x2 - x1)
        box_h = max(1, y2 - y1)
        aspect_ratio = box_w / float(box_h)

        # 1. Feeding zone check
        if feeding_zone and is_in_roi(bbox, feeding_zone, w, h):
            return BehaviorPrediction(
                behavior=BEHAVIOR_FEEDING,
                confidence=None,  # No fabricated confidence for heuristics
                mode="BASELINE_ESTIMATION",
                rationale="Animal detected within configured feeding bunk zone.",
            )

        # 2. Geometric posture analysis
        # In typical pen camera perspectives:
        # A recumbent/lying cow has a wide, low profile (aspect ratio > 1.70)
        # A standing cow has vertical limb clearance and lower aspect ratio (< 1.25)
        if aspect_ratio >= 1.75:
            return BehaviorPrediction(
                behavior=BEHAVIOR_LYING,
                confidence=None,
                mode="BASELINE_ESTIMATION",
                rationale=f"Recumbent posture estimated from wide horizontal profile (aspect ratio: {aspect_ratio:.2f}).",
            )
        elif aspect_ratio <= 1.20:
            return BehaviorPrediction(
                behavior=BEHAVIOR_STANDING,
                confidence=None,
                mode="BASELINE_ESTIMATION",
                rationale=f"Upright posture estimated from vertical profile (aspect ratio: {aspect_ratio:.2f}).",
            )

        # 3. Ambiguous posture -> Unknown
        return BehaviorPrediction(
            behavior=BEHAVIOR_UNKNOWN,
            confidence=None,
            mode="BASELINE_ESTIMATION",
            rationale=f"Posture ambiguous (aspect ratio: {aspect_ratio:.2f}). Marked Unknown to avoid false diagnosis.",
        )

    @staticmethod
    def _normalize_class_name(raw_name: str) -> str:
        """Maps diverse training labels to the 4 canonical behavior classes."""
        clean = raw_name.strip().lower()
        if "feed" in clean or "eat" in clean:
            return BEHAVIOR_FEEDING
        elif "ly" in clean or "rest" in clean or "sleep" in clean or "recumbent" in clean:
            return BEHAVIOR_LYING
        elif "stand" in clean or "walk" in clean or "move" in clean:
            return BEHAVIOR_STANDING
        return BEHAVIOR_UNKNOWN
