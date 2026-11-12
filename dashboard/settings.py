"""
Settings & System Configuration Dashboard View
AI-Based Real-Time Livestock Behavior & Inactivity Monitoring System
"""

import streamlit as st
from config import ProcessingConfig, CUSTOM_BEHAVIOR_MODEL_PATH
from database.database_operations import DatabaseManager
from utils.validators import validate_processing_parameters, validate_feeding_zone


def render_settings(config: ProcessingConfig, db_manager: DatabaseManager) -> None:
    """Renders parameter adjustments, model configuration, and database administration."""
    st.subheader("System Settings & Threshold Configuration")

    st.markdown("#### Inactivity & Welfare Thresholds")
    demo_mode = st.toggle(
        "Demonstration Mode (Fast Alerts)",
        value=config.demo_mode,
        help="When enabled, sets continuous lying threshold to 10 seconds for quick college lab demonstrations.",
    )
    config.demo_mode = demo_mode

    col_thresh1, col_thresh2 = st.columns(2)
    with col_thresh1:
        custom_thresh = st.number_input(
            "Standard Inactivity Threshold (seconds)",
            min_value=5.0,
            max_value=3600.0,
            value=float(config.inactivity_threshold_seconds),
            step=30.0,
            help="Default is 900s (15 minutes). Cow resting continuously past this triggers Critical alert.",
        )
        config.inactivity_threshold_seconds = custom_thresh

    with col_thresh2:
        cooldown = st.number_input(
            "Alert Cooldown (seconds)",
            min_value=5.0,
            max_value=600.0,
            value=float(config.alert_cooldown_seconds),
            step=5.0,
            help="Minimum seconds before repeating an alert for the same animal.",
        )
        config.alert_cooldown_seconds = cooldown

    st.markdown("---")
    st.markdown("#### Computer Vision & Detection Parameters")

    col_cv1, col_cv2 = st.columns(2)
    with col_cv1:
        conf = st.slider(
            "Cow Detection Confidence Threshold",
            min_value=0.10,
            max_value=0.90,
            value=float(config.confidence_threshold),
            step=0.05,
        )
        config.confidence_threshold = conf

    with col_cv2:
        frame_int = st.slider(
            "Frame Processing Interval (Skip rate)",
            min_value=1,
            max_value=10,
            value=int(config.frame_interval),
            help="1 = process every frame. 2 = process every 2nd frame (faster playback on older laptops).",
        )
        config.frame_interval = frame_int

    grace_period = st.slider(
        "Tracking Grace Period (seconds)",
        min_value=1.0,
        max_value=10.0,
        value=float(config.tracking_grace_period_seconds),
        step=0.5,
        help="Time an animal can be temporarily occluded before being marked Lost.",
    )
    config.tracking_grace_period_seconds = grace_period

    st.markdown("---")
    st.markdown("#### Custom AI Behavior Model")
    model_path_input = st.text_input(
        "Custom Behavior Model Path (.pt)",
        value=config.custom_model_path,
        help="Path to trained YOLO behavior weights. If absent, system operates in transparent Baseline Mode.",
    )
    config.custom_model_path = model_path_input

    st.markdown("---")
    st.markdown("#### Feeding Bunk Region of Interest (Normalized 0.0 - 1.0)")
    enable_zone = st.checkbox("Enable Ground Feeding Bunk Zone", value=bool(config.feeding_zone))

    if enable_zone:
        c_z1, c_z2, c_z3, c_z4 = st.columns(4)
        xmin = c_z1.number_input("xmin", 0.0, 1.0, float(config.feeding_zone.get("xmin", 0.0)), 0.05)
        ymin = c_z2.number_input("ymin", 0.0, 1.0, float(config.feeding_zone.get("ymin", 0.6)), 0.05)
        xmax = c_z3.number_input("xmax", 0.0, 1.0, float(config.feeding_zone.get("xmax", 1.0)), 0.05)
        ymax = c_z4.number_input("ymax", 0.0, 1.0, float(config.feeding_zone.get("ymax", 1.0)), 0.05)

        zone = {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}
        valid_zone, z_err = validate_feeding_zone(zone)
        if valid_zone:
            config.feeding_zone = zone
        else:
            st.warning(f"Invalid feeding zone: {z_err}")
    else:
        config.feeding_zone = {}

    st.markdown("---")
    st.markdown("#### Database Maintenance")
    st.warning("Resetting the database will purge all historical activity logs, tracked animals, and alerts.")
    if st.button("Reset / Clear Database Records", type="secondary"):
        db_manager.clear_database()
        st.success("Database cleared successfully for new clean demonstration session.")
        st.rerun()
