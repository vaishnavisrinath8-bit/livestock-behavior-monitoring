"""
Live Monitoring Dashboard View
AI-Based Real-Time Livestock Behavior & Inactivity Monitoring System
"""

import os
import time
from pathlib import Path
from typing import Optional
import streamlit as st
import cv2
import pandas as pd

from config import UPLOADS_DIR, SAMPLE_VIDEOS_DIR, SUPPORTED_VIDEO_EXTENSIONS, ProcessingConfig
from utils.video_utils import validate_video_file, get_video_metadata
from utils.image_utils import bgr_to_rgb
from utils.time_utils import seconds_to_hms
from processing.video_processor import VideoProcessor
from database.database_operations import DatabaseManager


def render_live_monitoring(
    video_processor: VideoProcessor,
    db_manager: DatabaseManager,
    config: ProcessingConfig,
) -> None:
    """Renders interactive video upload, webcam surveillance, and real-time processing feed."""
    st.subheader("Live Livestock Surveillance & Inactivity Tracking")

    # Mode selection: Upload vs Webcam
    input_source = st.radio(
        "Select Video Input Mode",
        options=["Uploaded Video File", "Live Webcam Feed", "Sample Videos"],
        horizontal=True,
    )

    video_path_to_process: Optional[Path] = None

    if input_source == "Uploaded Video File":
        uploaded_file = st.file_uploader(
            "Upload Cattle Video Footage",
            type=["mp4", "avi", "mov"],
            help="Upload an MP4, AVI, or MOV video clip of livestock pen or pasture.",
        )
        if uploaded_file:
            save_path = UPLOADS_DIR / uploaded_file.name
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            video_path_to_process = save_path

            # Show metadata
            meta = get_video_metadata(save_path)
            if meta:
                c1, c2, c3, c4 = st.columns(4)
                c1.caption(f"**Resolution:** {meta['width']}x{meta['height']}")
                c2.caption(f"**FPS:** {meta['fps']}")
                c3.caption(f"**Total Frames:** {meta['frame_count']}")
                c4.caption(f"**Duration:** {meta['duration_formatted']}")

    elif input_source == "Sample Videos":
        sample_files = [f for f in SAMPLE_VIDEOS_DIR.glob("*") if f.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS]
        if sample_files:
            chosen = st.selectbox("Choose a sample video", options=[f.name for f in sample_files])
            video_path_to_process = SAMPLE_VIDEOS_DIR / chosen
            meta = get_video_metadata(video_path_to_process)
            if meta:
                st.caption(f"Video: {meta['width']}x{meta['height']} @ {meta['fps']} FPS ({meta['duration_formatted']})")
        else:
            st.info(
                f"No sample videos found in `{SAMPLE_VIDEOS_DIR}`. "
                "You can upload your own video footage using the Upload tab."
            )

    # Inactivity Threshold Quick Indicator
    eff_thresh = config.get_effective_inactivity_threshold()
    mode_tag = "Demonstration Mode (10s)" if config.demo_mode else f"Standard Mode ({seconds_to_hms(eff_thresh)})"
    st.caption(f"**Active Inactivity Threshold:** {mode_tag} &bull; Adjust in **Settings**.")

    col_btn_start, col_btn_stop, _ = st.columns([1, 1, 3])

    if "is_processing" not in st.session_state:
        st.session_state["is_processing"] = False
    if "stop_requested" not in st.session_state:
        st.session_state["stop_requested"] = False

    with col_btn_start:
        start_btn = st.button("Start Monitoring", type="primary", use_container_width=True)
    with col_btn_stop:
        stop_btn = st.button("Stop Monitoring", use_container_width=True)

    if stop_btn:
        st.session_state["stop_requested"] = True
        st.session_state["is_processing"] = False
        st.warning("Monitoring stopped by user.")

    if start_btn:
        st.session_state["stop_requested"] = False
        st.session_state["is_processing"] = True

    # Processing Loop
    if st.session_state["is_processing"]:
        # Prepare layout containers
        prog_bar = st.progress(0.0)
        status_text = st.empty()

        col_feed, col_side = st.columns([3, 2])
        with col_feed:
            video_placeholder = st.empty()
        with col_side:
            metrics_placeholder = st.empty()
            table_placeholder = st.empty()

        def should_stop() -> bool:
            return st.session_state.get("stop_requested", False)

        try:
            if input_source == "Live Webcam Feed":
                stream = video_processor.process_webcam(camera_index=0, stop_check=should_stop)
            else:
                if not video_path_to_process or not video_path_to_process.exists():
                    st.error("Please select or upload a valid video file to start processing.")
                    st.session_state["is_processing"] = False
                    return

                def update_progress(pct: float, cur_frame: int, total_frames: int):
                    prog_bar.progress(pct)
                    status_text.text(f"Processing Frame {cur_frame}/{total_frames} ({pct*100:.1f}%)")

                stream = video_processor.process_file(
                    video_path=video_path_to_process,
                    stop_check=should_stop,
                    progress_callback=update_progress,
                )

            for result in stream:
                if should_stop():
                    break

                # Convert annotated frame to RGB for Streamlit rendering
                rgb_frame = bgr_to_rgb(result.annotated_frame)
                video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

                # Render active metrics
                with metrics_placeholder.container():
                    m1, m2 = st.columns(2)
                    m1.metric("Cows in Frame", result.cows_count)
                    m2.metric("Timestamp", seconds_to_hms(result.timestamp_seconds))

                # Render active tracking table
                with table_placeholder.container():
                    st.markdown("##### Current Animal States")
                    if result.active_states:
                        rows = []
                        for st_item in result.active_states:
                            rows.append({
                                "Cow #": f"#{st_item.tracking_id}",
                                "Behavior": st_item.current_behavior,
                                "Lying Time": seconds_to_hms(st_item.current_lying_duration),
                                "Total Time": seconds_to_hms(st_item.total_observed_duration),
                                "Status": st_item.alert_severity,
                            })
                        df_active = pd.DataFrame(rows)
                        st.dataframe(df_active, hide_index=True, use_container_width=True)
                    else:
                        st.caption("No cows detected in current frame.")

            st.session_state["is_processing"] = False
            status_text.text("Monitoring session completed or stopped.")
            st.success("Video processing finished. All observations saved to SQLite.")

        except Exception as e:
            st.session_state["is_processing"] = False
            st.error(f"Processing halted: {e}")
