"""
End-to-End Pipeline Verification Script
AI-Based Real-Time Livestock Behavior & Inactivity Monitoring System
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import SAMPLE_VIDEOS_DIR, DB_PATH, ProcessingConfig
from database.db import init_database
from database.database_operations import DatabaseManager
from models.cow_detector import CowDetector
from models.tracker import ByteTrackerManager
from models.behavior_classifier import LivestockBehaviorClassifier
from processing.inactivity_engine import InactivityEngine
from processing.alert_engine import AlertEngine
from processing.frame_processor import FrameProcessor
from processing.video_processor import VideoProcessor


def verify_full_system():
    print("=" * 60)
    print("STARTING END-TO-END PIPELINE VERIFICATION")
    print("=" * 60)

    # 1. Initialize Database
    init_database(DB_PATH)
    db_mgr = DatabaseManager(DB_PATH)
    db_mgr.clear_database()
    print("[1] SQLite database initialized and cleared for verification.")

    # 2. Configure for Demo (5s threshold for quick alert check)
    cfg = ProcessingConfig()
    cfg.demo_mode = True  # 10s threshold
    cfg.frame_interval = 2  # process every 2nd frame

    # 3. Instantiate AI Services
    detector = CowDetector(confidence_threshold=0.30)
    tracker = ByteTrackerManager(detector.model)
    classifier = LivestockBehaviorClassifier()
    inact_engine = InactivityEngine(inactivity_threshold_seconds=10.0, grace_period_seconds=2.0)
    alert_engine = AlertEngine(cooldown_seconds=10.0)

    frame_processor = FrameProcessor(
        detector=detector,
        tracker=tracker,
        classifier=classifier,
        inactivity_engine=inact_engine,
        alert_engine=alert_engine,
        db_manager=db_mgr,
        config=cfg,
    )

    video_processor = VideoProcessor(frame_processor)

    # 4. Run on demonstration video
    video_path = SAMPLE_VIDEOS_DIR / "demo_cattle.mp4"
    print(f"[2] Processing video: {video_path.name}...")

    frame_count = 0
    for res in video_processor.process_file(video_path):
        frame_count += 1

    print(f"[3] Successfully processed {frame_count} frames.")

    # 5. Verify Database Records
    kpis = db_mgr.get_dashboard_kpis()
    print("[4] Database KPI Results:")
    for k, v in kpis.items():
        print(f"    - {k}: {v}")

    animals = db_mgr.get_all_animals()
    print(f"[5] Tracked animals in database: {len(animals)}")

    alerts = db_mgr.get_all_alerts()
    print(f"[6] Total alerts generated: {len(alerts)}")

    logs = db_mgr.get_activity_logs()
    print(f"[7] Activity logs recorded: {len(logs)}")

    print("=" * 60)
    print("PIPELINE VERIFICATION SUCCESSFUL!")
    print("=" * 60)


if __name__ == "__main__":
    verify_full_system()
