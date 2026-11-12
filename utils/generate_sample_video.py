"""
Sample Video Generator Using Photographic Cattle Data
AI-Based Real-Time Livestock Behavior & Inactivity Monitoring System
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np
from config import SAMPLE_VIDEOS_DIR, DATA_DIR


def generate_demonstration_video(
    output_path: Path = SAMPLE_VIDEOS_DIR / "demo_cattle.mp4",
    duration_seconds: int = 12,
    fps: int = 15,
    width: int = 800,
    height: int = 450,
) -> Path:
    """
    Composites photographic cattle footage to create a demonstration video
    that genuinely activates YOLOv8n COCO cow detection, ByteTrack tracking,
    and inactivity timers.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_frames = duration_seconds * fps

    # Load test cow image
    cow_img_path = DATA_DIR / "test_cow.jpg"
    if not cow_img_path.exists():
        raise FileNotFoundError(f"Source cow photo not found at {cow_img_path}")

    cow_src = cv2.imread(str(cow_img_path))
    src_h, src_w = cow_src.shape[:2]

    # Crop cow region
    # From previous detection: [62, 12, 393, 303]
    cow_crop = cow_src[12:303, 62:393]
    crop_h, crop_w = cow_crop.shape[:2]

    # Standing cow target size
    stand_w = 220
    stand_h = int(crop_h * (stand_w / float(crop_w)))
    stand_resized = cv2.resize(cow_crop, (stand_w, stand_h), interpolation=cv2.INTER_AREA)

    # Lying cow (wider aspect ratio, lower to ground)
    lying_w = 250
    lying_h = int(stand_h * 0.55)  # recumbent flat profile
    lying_resized = cv2.resize(cow_crop, (lying_w, lying_h), interpolation=cv2.INTER_AREA)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, float(fps), (width, height))

    for frame_idx in range(total_frames):
        t = frame_idx / float(fps)

        # Background: Pasture meadow
        frame = np.full((height, width, 3), (120, 180, 130), dtype=np.uint8)  # grass green
        # Barn wall / horizon at top
        cv2.rectangle(frame, (0, 0), (width, 140), (200, 210, 205), -1)
        cv2.line(frame, (0, 140), (width, 140), (100, 110, 105), 3)

        # Feeding bunk zone at bottom right
        cv2.rectangle(frame, (int(width * 0.65), int(height * 0.65)), (width - 20, height - 20), (160, 170, 165), -1)
        cv2.putText(
            frame,
            "FEEDING ZONE",
            (int(width * 0.67), int(height * 0.75)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (80, 90, 85),
            1,
            cv2.LINE_AA,
        )

        # 1. Cow A (Standing / Moving slowly left to right)
        cow_a_x = int(60 + 20 * np.sin(t * 0.8))
        cow_a_y = 170
        frame[cow_a_y : cow_a_y + stand_h, cow_a_x : cow_a_x + stand_w] = stand_resized

        # 2. Cow B (Lying / Resting continuously in the field)
        cow_b_x = 420
        cow_b_y = 230
        frame[cow_b_y : cow_b_y + lying_h, cow_b_x : cow_b_x + lying_w] = lying_resized

        # Overlay timestamp banner
        cv2.putText(
            frame,
            f"Livestock Barn Camera - Playback: {t:.1f}s | Frame {frame_idx + 1}/{total_frames}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (40, 50, 40),
            2,
            cv2.LINE_AA,
        )

        out.write(frame)

    out.release()
    print(f"Photographic livestock demonstration video created: {output_path}")
    return output_path


if __name__ == "__main__":
    generate_demonstration_video()
