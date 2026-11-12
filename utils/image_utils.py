"""
Image Preprocessing and Annotation Utilities
AI-Based Real-Time Livestock Behavior & Inactivity Monitoring System
"""

from typing import Tuple, Optional, Dict
import cv2
import numpy as np


def resize_frame(frame: np.ndarray, target_width: int = 960) -> np.ndarray:
    """
    Resizes a frame proportionally to target_width to ensure consistent
    and fast processing on laptop hardware without distorting aspect ratio.
    """
    if frame is None or frame.size == 0:
        return frame
    height, width = frame.shape[:2]
    if width <= target_width:
        return frame
    aspect_ratio = height / width
    target_height = int(target_width * aspect_ratio)
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    """Converts OpenCV BGR image array to RGB for PIL / Streamlit display."""
    if frame is None:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(frame: np.ndarray) -> np.ndarray:
    """Converts RGB image array to OpenCV BGR format."""
    if frame is None:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def is_in_roi(
    bbox: Tuple[int, int, int, int],
    roi: Dict[str, float],
    frame_width: int,
    frame_height: int,
) -> bool:
    """
    Determines if the center bottom (feet position) of a bounding box
    falls inside a designated normalized Region of Interest (e.g. feeding bunk).
    roi keys: 'xmin', 'ymin', 'xmax', 'ymax' in range [0.0, 1.0].
    """
    if not roi or not all(k in roi for k in ("xmin", "ymin", "xmax", "ymax")):
        return False

    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0 / frame_width
    cy_bottom = y2 / frame_height  # Feet location is most informative for ground zones

    return (
        roi["xmin"] <= cx <= roi["xmax"]
        and roi["ymin"] <= cy_bottom <= roi["ymax"]
    )


def draw_bounding_box(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    tracking_id: Optional[int],
    behavior: str,
    color: Tuple[int, int, int],
    confidence: Optional[float] = None,
    alert_banner: Optional[str] = None,
) -> np.ndarray:
    """
    Draws professional bounding boxes and information chips over detected animals.
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = frame.shape[:2]

    # Clip to frame boundary
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)

    # Main detection rectangle
    thickness = 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Corner brackets for modern CV aesthetic
    line_len = min(20, int((x2 - x1) * 0.25), int((y2 - y1) * 0.25))
    if line_len > 4:
        # Top-left
        cv2.line(frame, (x1, y1), (x1 + line_len, y1), color, thickness + 2)
        cv2.line(frame, (x1, y1), (x1, y1 + line_len), color, thickness + 2)
        # Top-right
        cv2.line(frame, (x2, y1), (x2 - line_len, y1), color, thickness + 2)
        cv2.line(frame, (x2, y1), (x2, y1 + line_len), color, thickness + 2)
        # Bottom-left
        cv2.line(frame, (x1, y2), (x1 + line_len, y2), color, thickness + 2)
        cv2.line(frame, (x1, y2), (x1, y2 - line_len), color, thickness + 2)
        # Bottom-right
        cv2.line(frame, (x2, y2), (x2 - line_len, y2), color, thickness + 2)
        cv2.line(frame, (x2, y2), (x2, y2 - line_len), color, thickness + 2)

    # Label text
    id_str = f"Cow #{tracking_id}" if tracking_id is not None else "Cow"
    conf_str = f" {confidence:.2f}" if confidence is not None else ""
    label_text = f"{id_str} | {behavior}{conf_str}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.52
    font_thickness = 1

    (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)
    box_top = max(0, y1 - text_h - baseline - 8)
    box_bottom = y1

    # Filled background for high readability
    cv2.rectangle(frame, (x1, box_top), (x1 + text_w + 12, box_bottom), color, -1)
    # White text
    cv2.putText(
        frame,
        label_text,
        (x1 + 6, box_bottom - baseline - 2),
        font,
        font_scale,
        (255, 255, 255),
        font_thickness,
        cv2.LINE_AA,
    )

    # Optional Alert Warning Tag
    if alert_banner:
        alert_h = 20
        banner_top = max(0, box_top - alert_h - 4)
        cv2.rectangle(frame, (x1, banner_top), (x1 + 220, box_top - 2), (0, 0, 220), -1)
        cv2.putText(
            frame,
            f"! {alert_banner}",
            (x1 + 6, box_top - 6),
            font,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return frame
