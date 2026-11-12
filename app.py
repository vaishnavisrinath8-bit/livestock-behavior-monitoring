"""
Main Application Entry Point
AI-Based Real-Time Livestock Behavior & Inactivity Monitoring System
"""

import sys
import logging
from pathlib import Path
import streamlit as st

# Configure Root Logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("livestock_app")

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import (
    DB_PATH,
    DEFAULT_DETECTION_MODEL,
    ProcessingConfig,
)
from database.db import init_database
from database.database_operations import DatabaseManager
from models.cow_detector import CowDetector
from models.tracker import ByteTrackerManager
from models.behavior_classifier import LivestockBehaviorClassifier
from processing.inactivity_engine import InactivityEngine
from processing.alert_engine import AlertEngine
from processing.frame_processor import FrameProcessor
from processing.video_processor import VideoProcessor

from dashboard.sidebar import render_sidebar
from dashboard.overview import render_overview
from dashboard.live_monitoring import render_live_monitoring
from dashboard.animal_details import render_animal_tracking, render_animal_details
from dashboard.analytics import render_analytics
from dashboard.alerts import render_alerts
from dashboard.activity_history import render_activity_history
from dashboard.settings import render_settings


def setup_streamlit_page():
    """Configures page title, layout, and custom modern farm styling."""
    st.set_page_config(
        page_title="LIVESTOCK AI MONITORING CENTER",
        page_icon="🐄",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Professional clean styling in dark green and white
    st.markdown(
        """
        <style>
            /* Main container padding */
            .block-container {
                padding-top: 1.8rem;
                padding-bottom: 2.5rem;
                padding-left: 2.5rem;
                padding-right: 2.5rem;
            }
            /* Brand headers */
            h1, h2, h3 {
                color: #1b5e20;
                font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
            }
            /* Metric cards styling */
            [data-testid="stMetric"] {
                background-color: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 10px;
                padding: 14px 18px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.04);
            }
            [data-testid="stMetricLabel"] {
                color: #555555;
                font-weight: 600;
                font-size: 0.85rem;
            }
            [data-testid="stMetricValue"] {
                color: #1b5e20;
                font-weight: 700;
            }
            /* Buttons */
            .stButton>button {
                border-radius: 6px;
                font-weight: 600;
            }
            /* Alerts box override */
            div[data-testid="stAlert"] {
                border-radius: 8px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialize_runtime_services():
    """Initializes and caches models, engines, and database connections."""
    # 1. Initialize SQLite Database Tables
    init_database(DB_PATH)

    if "db_manager" not in st.session_state:
        st.session_state["db_manager"] = DatabaseManager(DB_PATH)

    if "config" not in st.session_state:
        st.session_state["config"] = ProcessingConfig()

    cfg: ProcessingConfig = st.session_state["config"]

    if "detector" not in st.session_state:
        st.session_state["detector"] = CowDetector(
            model_name_or_path=DEFAULT_DETECTION_MODEL,
            confidence_threshold=cfg.confidence_threshold,
            iou_threshold=cfg.iou_threshold,
        )

    if "tracker" not in st.session_state:
        st.session_state["tracker"] = ByteTrackerManager(st.session_state["detector"].model)

    if "classifier" not in st.session_state:
        st.session_state["classifier"] = LivestockBehaviorClassifier(
            model_path=Path(cfg.custom_model_path)
        )

    if "inactivity_engine" not in st.session_state:
        st.session_state["inactivity_engine"] = InactivityEngine(
            inactivity_threshold_seconds=cfg.get_effective_inactivity_threshold(),
            grace_period_seconds=cfg.tracking_grace_period_seconds,
        )

    if "alert_engine" not in st.session_state:
        st.session_state["alert_engine"] = AlertEngine(
            cooldown_seconds=cfg.alert_cooldown_seconds
        )

    if "frame_processor" not in st.session_state:
        st.session_state["frame_processor"] = FrameProcessor(
            detector=st.session_state["detector"],
            tracker=st.session_state["tracker"],
            classifier=st.session_state["classifier"],
            inactivity_engine=st.session_state["inactivity_engine"],
            alert_engine=st.session_state["alert_engine"],
            db_manager=st.session_state["db_manager"],
            config=cfg,
        )

    if "video_processor" not in st.session_state:
        st.session_state["video_processor"] = VideoProcessor(
            frame_processor=st.session_state["frame_processor"]
        )


def main():
    setup_streamlit_page()
    initialize_runtime_services()

    db_manager: DatabaseManager = st.session_state["db_manager"]
    config: ProcessingConfig = st.session_state["config"]
    video_processor: VideoProcessor = st.session_state["video_processor"]

    selected_view = render_sidebar()

    if selected_view == "Dashboard Overview":
        render_overview(db_manager)
    elif selected_view == "Live Monitoring":
        render_live_monitoring(video_processor, db_manager, config)
    elif selected_view == "Animal Tracking":
        render_animal_tracking(db_manager)
    elif selected_view == "Animal Details":
        render_animal_details(db_manager)
    elif selected_view == "Behavior Analytics":
        render_analytics(db_manager)
    elif selected_view == "Inactivity Alerts":
        render_alerts(db_manager)
    elif selected_view == "Activity History":
        render_activity_history(db_manager)
    elif selected_view == "Settings":
        render_settings(config, db_manager)


if __name__ == "__main__":
    main()
