"""
Video and Stream Processing Orchestration
AI-Based Real-Time Livestock Behavior & Inactivity Monitoring System
"""

import time
import logging
from pathlib import Path
from typing import Generator, Optional, Callable, Dict, Any, Union
import cv2

from config import ProcessingConfig
from utils.video_utils import validate_video_file, get_video_metadata
from utils.time_utils import calculate_video_timestamp
from processing.frame_processor import FrameProcessor, ProcessedFrameResult

logger = logging.getLogger(__name__)


class VideoProcessor:
    """
    Orchestrates continuous frame extraction and sequential processing
    for uploaded video files and local webcam feeds.
    """

    def __init__(self, frame_processor: FrameProcessor):
        self.frame_processor = frame_processor

    def process_file(
        self,
        video_path: Union[str, Path],
        stop_check: Optional[Callable[[], bool]] = None,
        progress_callback: Optional[Callable[[float, int, int], None]] = None,
    ) -> Generator[ProcessedFrameResult, None, None]:
        """
        Sequentially reads and processes an uploaded video file.
        Uses video FPS to compute accurate timestamps independent of inference speed.
        """
        path = Path(video_path)
        is_valid, msg = validate_video_file(path)
        if not is_valid:
            logger.error("Video validation failed: %s", msg)
            raise ValueError(msg)

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {path.name}")

        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            if fps <= 0 or fps > 120:
                fps = 25.0  # Sensible standard fallback

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_idx = 0
            processed_idx = 0
            frame_interval = max(1, self.frame_processor.config.frame_interval)

            while True:
                # User-requested cancellation check
                if stop_check and stop_check():
                    logger.info("Processing stopped by user request at frame %d.", frame_idx)
                    break

                ret, frame = cap.read()
                if not ret or frame is None:
                    logger.info("Reached end of video stream at frame %d.", frame_idx)
                    break

                frame_idx += 1

                # Frame skipping logic if configured
                if frame_idx % frame_interval != 0:
                    continue

                processed_idx += 1
                # Exact video playback timestamp
                timestamp_sec = calculate_video_timestamp(frame_idx, fps)

                # Execute full AI monitoring pipeline
                result = self.frame_processor.process_frame(
                    frame=frame,
                    frame_number=frame_idx,
                    timestamp_seconds=timestamp_sec,
                    source_type="video",
                )

                if progress_callback and total_frames > 0:
                    pct = min(1.0, frame_idx / float(total_frames))
                    progress_callback(pct, frame_idx, total_frames)

                yield result

        except Exception as e:
            logger.error("Exception during video stream processing: %s", e)
            raise
        finally:
            cap.release()
            logger.info("Video capture released successfully.")

    def process_webcam(
        self,
        camera_index: int = 0,
        stop_check: Optional[Callable[[], bool]] = None,
    ) -> Generator[ProcessedFrameResult, None, None]:
        """
        Streams frames in real-time from a connected webcam.
        Uses wall-clock elapsed time for real-time webcam monitoring.
        """
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            logger.error("Webcam with index %d is not available or locked.", camera_index)
            raise RuntimeError(f"Webcam (Device {camera_index}) is not accessible. Please check camera connection and permissions.")

        start_wall_time = time.time()
        frame_idx = 0

        try:
            while True:
                if stop_check and stop_check():
                    logger.info("Webcam monitoring stopped by user request.")
                    break

                ret, frame = cap.read()
                if not ret or frame is None:
                    logger.warning("Failed to grab frame from webcam stream.")
                    time.sleep(0.05)
                    continue

                frame_idx += 1
                elapsed_wall_time = time.time() - start_wall_time

                result = self.frame_processor.process_frame(
                    frame=frame,
                    frame_number=frame_idx,
                    timestamp_seconds=elapsed_wall_time,
                    source_type="webcam",
                )

                yield result

        finally:
            cap.release()
            logger.info("Webcam capture released successfully.")
