"""
Animal Tracking & Individual Details Dashboard View
AI-Based Real-Time Livestock Behavior & Inactivity Monitoring System
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from database.database_operations import DatabaseManager
from config import BEHAVIOR_COLORS_HEX, BEHAVIOR_STANDING, BEHAVIOR_FEEDING, BEHAVIOR_LYING, BEHAVIOR_UNKNOWN
from utils.time_utils import seconds_to_hms


def render_animal_tracking(db_manager: DatabaseManager) -> None:
    """Renders the comprehensive animal tracking inventory table."""
    st.subheader("Individual Animal Tracking Registry")

    status_filter = st.selectbox(
        "Filter by Monitoring Status",
        options=["All Animals", "Active", "Lost"],
        index=0,
    )
    filter_val = None if status_filter == "All Animals" else status_filter

    animals = db_manager.get_all_animals(status=filter_val)
    if not animals:
        st.info("No animal tracking records found matching the filter.")
        return

    df = pd.DataFrame(animals)
    df_display = pd.DataFrame()
    df_display["Animal ID"] = df["tracking_id"].apply(lambda x: f"Cow #{x}")
    df_display["Current Behavior"] = df["current_behavior"]
    df_display["Total Observation Time"] = df["total_observed_seconds"].apply(seconds_to_hms)
    df_display["First Seen (UTC)"] = df["first_seen"]
    df_display["Last Seen (UTC)"] = df["last_seen"]
    df_display["Monitoring Status"] = df["status"]

    st.dataframe(df_display, use_container_width=True, hide_index=True)


def render_animal_details(db_manager: DatabaseManager) -> None:
    """Renders exhaustive historical and behavioral profile for a single tracked animal."""
    st.subheader("Individual Animal Profile & Welfare History")

    animals = db_manager.get_all_animals()
    if not animals:
        st.info("No animal records available yet. Run Live Monitoring on a video first.")
        return

    animal_options = {f"Cow #{a['tracking_id']} ({a['status']})": a["tracking_id"] for a in animals}
    selected_label = st.selectbox("Select Tracked Animal", options=list(animal_options.keys()))
    selected_id = animal_options[selected_label]

    animal = db_manager.get_animal(selected_id)
    if not animal:
        st.error("Could not find record for selected animal.")
        return

    # Top Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Animal ID", f"Cow #{animal['tracking_id']}")
    with c2:
        st.metric("Current Behavior", animal["current_behavior"])
    with c3:
        st.metric("Total Observed", seconds_to_hms(animal["total_observed_seconds"]))
    with c4:
        st.metric("Status", animal["status"])

    st.markdown("---")

    # Behavior Duration Breakdown
    durations = db_manager.get_behavior_duration_by_animal(selected_id)
    c_left, c_right = st.columns([1, 1])

    with c_left:
        st.markdown("##### Behavior Duration Breakdown")
        if durations and any(v > 0 for v in durations.values()):
            df_dur = pd.DataFrame(
                [{"Behavior": k, "Duration (s)": v, "Formatted": seconds_to_hms(v)} for k, v in durations.items()]
            )
            fig = px.bar(
                df_dur,
                x="Behavior",
                y="Duration (s)",
                color="Behavior",
                color_discrete_map=BEHAVIOR_COLORS_HEX,
                text="Formatted",
            )
            fig.update_layout(height=280, margin=dict(l=10, r=10, t=20, b=10), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("No duration events completed yet for this animal.")

    with c_right:
        st.markdown("##### Alert History for this Animal")
        all_alerts = db_manager.get_all_alerts(limit=50)
        animal_alerts = [a for a in all_alerts if a["tracking_id"] == selected_id]

        if animal_alerts:
            df_a = pd.DataFrame(animal_alerts)[["id", "severity", "inactivity_seconds", "message", "created_at", "resolved"]]
            df_a["inactivity_seconds"] = df_a["inactivity_seconds"].apply(seconds_to_hms)
            df_a["resolved"] = df_a["resolved"].apply(lambda r: "Resolved" if r == 1 else "Active")
            df_a.columns = ["Alert ID", "Severity", "Duration", "Message", "Timestamp", "Status"]
            st.dataframe(df_a, use_container_width=True, hide_index=True)
        else:
            st.success("No inactivity alerts logged for this animal.")
