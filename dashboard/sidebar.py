"""
Dashboard Sidebar and Global Navigation Component
AI-Based Real-Time Livestock Behavior & Inactivity Monitoring System
"""

import streamlit as st
from config import DASHBOARD_THEME


def render_sidebar() -> str:
    """
    Renders professional sidebar with brand identity, system status, and page selection.
    Returns the key of the selected view.
    """
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 10px 0 15px 0; border-bottom: 1px solid #e0e0e0; margin-bottom: 15px;">
                <h2 style="color: #1b5e20; margin: 0; font-size: 1.35rem; font-weight: 700; letter-spacing: -0.5px;">
                    LIVESTOCK AI
                </h2>
                <p style="color: #666; margin: 4px 0 0 0; font-size: 0.82rem; font-weight: 500;">
                    Monitoring Center
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        pages = [
            "Dashboard Overview",
            "Live Monitoring",
            "Animal Tracking",
            "Animal Details",
            "Behavior Analytics",
            "Inactivity Alerts",
            "Activity History",
            "Settings",
        ]

        # Use query parameter or session state to maintain selection
        selected_page = st.radio(
            "Navigation",
            options=pages,
            index=0,
            label_visibility="collapsed",
        )

        st.markdown("---")

        # System Status Panel in Sidebar
        st.markdown(
            """
            <div style="background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; margin-top: 10px;">
                <div style="font-size: 0.78rem; font-weight: 700; color: #1b5e20; text-transform: uppercase; margin-bottom: 6px;">
                    System Status
                </div>
                <div style="font-size: 0.8rem; color: #333; margin-bottom: 4px;">
                    <b>Detector:</b> YOLOv8n (COCO Cow)
                </div>
                <div style="font-size: 0.8rem; color: #333; margin-bottom: 4px;">
                    <b>Tracker:</b> ByteTrack
                </div>
                <div style="font-size: 0.8rem; color: #333; margin-bottom: 4px;">
                    <b>Classifier:</b> Modular Engine
                </div>
                <div style="font-size: 0.8rem; color: #333;">
                    <b>Storage:</b> SQLite WAL
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div style="margin-top: 25px; font-size: 0.75rem; color: #888; text-align: center;">
                v1.0.0 &bull; Local Execution Mode
            </div>
            """,
            unsafe_allow_html=True,
        )

    return selected_page
