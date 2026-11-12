"""
Activity History Dashboard View
AI-Based Real-Time Livestock Behavior & Inactivity Monitoring System
"""

import streamlit as st
import pandas as pd
from database.database_operations import DatabaseManager
from utils.time_utils import seconds_to_hms


def render_activity_history(db_manager: DatabaseManager) -> None:
    """Renders chronological behavior activity log events with animal and date filters."""
    st.subheader("Historical Activity & Behavior Logs")

    animals = db_manager.get_all_animals()
    animal_choices = ["All Animals"] + [f"Cow #{a['tracking_id']}" for a in animals]

    col_filter1, col_filter2 = st.columns(2)
    with col_filter1:
        selected_cow_label = st.selectbox("Filter by Animal", options=animal_choices)
    with col_filter2:
        limit = st.slider("Record Limit", min_value=25, max_value=500, value=100, step=25)

    filter_id = None
    if selected_cow_label != "All Animals":
        filter_id = int(selected_cow_label.replace("Cow #", ""))

    logs = db_manager.get_activity_logs(tracking_id=filter_id, limit=limit)

    if not logs:
        st.info("No behavior activity logs found matching the filter.")
        return

    df = pd.DataFrame(logs)
    df_display = pd.DataFrame()
    df_display["Log ID"] = df["id"]
    df_display["Animal"] = df["tracking_id"].apply(lambda x: f"Cow #{x}")
    df_display["Behavior"] = df["behavior"]
    df_display["Duration"] = df["duration_seconds"].apply(seconds_to_hms)
    df_display["Timestamp (UTC)"] = df["timestamp"]
    df_display["Source"] = df["source_type"]

    st.dataframe(df_display, use_container_width=True, hide_index=True)
