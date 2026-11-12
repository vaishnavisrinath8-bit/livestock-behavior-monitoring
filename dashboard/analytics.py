"""
Behavior Analytics Dashboard View
AI-Based Real-Time Livestock Behavior & Inactivity Monitoring System
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from database.database_operations import DatabaseManager
from config import BEHAVIOR_COLORS_HEX
from utils.time_utils import seconds_to_hms


def render_analytics(db_manager: DatabaseManager) -> None:
    """Renders advanced population-level behavior and inactivity analytics."""
    st.subheader("Population Behavior & Inactivity Analytics")

    animals = db_manager.get_all_animals()
    logs = db_manager.get_activity_logs(limit=500)

    if not animals or not logs:
        st.info("Insufficient monitoring data available for analytics. Run video processing to populate records.")
        return

    df_logs = pd.DataFrame(logs)
    df_animals = pd.DataFrame(animals)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### Cumulative Time per Behavior (All Animals)")
        durations = db_manager.get_aggregated_behavior_durations()
        if durations and any(v > 0 for v in durations.values()):
            df_pie = pd.DataFrame([{"Behavior": k, "Seconds": v} for k, v in durations.items()])
            fig_pie = px.pie(
                df_pie,
                names="Behavior",
                values="Seconds",
                color="Behavior",
                color_discrete_map=BEHAVIOR_COLORS_HEX,
                hole=0.4,
            )
            fig_pie.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.caption("No duration logs available.")

    with col2:
        st.markdown("##### Total Observation Time by Animal")
        if not df_animals.empty:
            df_animals["Cow Label"] = df_animals["tracking_id"].apply(lambda x: f"Cow #{x}")
            fig_obs = px.bar(
                df_animals,
                x="Cow Label",
                y="total_observed_seconds",
                color="status",
                labels={"total_observed_seconds": "Observed Time (s)", "Cow Label": ""},
                color_discrete_map={"Active": "#28a745", "Lost": "#dc3545"},
            )
            fig_obs.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_obs, use_container_width=True)

    st.markdown("---")

    # Time series / event progression
    st.markdown("##### Behavior Events Log Progression")
    if not df_logs.empty and "timestamp" in df_logs.columns:
        fig_events = px.scatter(
            df_logs,
            x="timestamp",
            y="tracking_id",
            color="behavior",
            size="duration_seconds",
            color_discrete_map=BEHAVIOR_COLORS_HEX,
            labels={"tracking_id": "Cow ID", "timestamp": "Event Timestamp", "duration_seconds": "Duration (s)"},
        )
        fig_events.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_events, use_container_width=True)
