"""
Video File Validation and Metadata Utilities
AI-Based Real-Time Livestock Behavior & Inactivity Monitoring System
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import cv2
from config import SUPPORTED_VIDEO_EXTENSIONS


def validate_video_file(file_path: Path) -> Tuple[bool, str]:
    """
    Validates file existence, supported format extension, non-zero file size,
    and OpenCV stream readability.
    """
    path = Path(file_path)
    if not path.exists():
        return False, f"File does not exist: {path.name}"

    if path.stat().st_size == 0:
        return False, f"Video file is empty (0 bytes): {path.name}"

    ext = path.suffix.lower()
    if ext not in SUPPORTED_VIDEO_EXTENSIONS:
        return False, f"Unsupported format '{ext}'. Supported: {', '.join(SUPPORTED_VIDEO_EXTENSIONS)}"

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        return False, "Could not open video file. File may be corrupted or codec missing."

    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None or frame.size == 0:
        return False, "Cannot read initial frame from video. File may be incomplete or invalid."

    return True, "Valid video file"


def get_video_metadata(file_path: Path) -> Optional[Dict[str, Any]]:
    """
    Extracts frame count, FPS, duration, resolution, and codec information from a video.
    """
    cap = cv2.VideoCapture(str(file_path))
    if not cap.isOpened():
        cap.release()
        return None

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_seconds = frame_count / fps if fps > 0 and frame_count > 0 else 0.0

        return {
            "file_name": Path(file_path).name,
            "width": width,
            "height": height,
            "fps": round(fps, 2),
            "frame_count": frame_count,
            "duration_seconds": round(duration_seconds, 2),
            "duration_formatted": f"{int(duration_seconds // 60):02d}:{int(duration_seconds % 60):02d}",
        }
    finally:
        cap.release()
