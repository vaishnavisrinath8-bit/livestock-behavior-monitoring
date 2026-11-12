"""
Dashboard Overview View
AI-Based Real-Time Livestock Behavior & Inactivity Monitoring System
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from database.database_operations import DatabaseManager
from config import BEHAVIOR_COLORS_HEX, BEHAVIOR_STANDING, BEHAVIOR_FEEDING, BEHAVIOR_LYING, BEHAVIOR_UNKNOWN
from utils.time_utils import seconds_to_hms


def render_overview(db_manager: DatabaseManager) -> None:
    """Renders the top-level analytical overview with live metrics and charts."""
    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <h1 style="color: #1b5e20; margin: 0; font-size: 1.85rem; font-weight: 700;">
                LIVESTOCK AI MONITORING CENTER
            </h1>
            <p style="color: #555; margin: 4px 0 0 0; font-size: 0.95rem;">
                Intelligent Animal Behavior and Welfare Monitoring
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Database-driven KPIs
    kpis = db_manager.get_dashboard_kpis()

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric(label="Total Tracked", value=kpis["total_animals"])
    with c2:
        st.metric(label="Currently Active", value=kpis["active_animals"])
    with c3:
        st.metric(label="Feeding", value=kpis["feeding_count"])
    with c4:
        st.metric(label="Resting / Lying", value=kpis["resting_count"])
    with c5:
        alert_val = kpis["active_alerts"]
        st.metric(label="Active Alerts", value=alert_val, delta=f"{alert_val} Unresolved" if alert_val > 0 else "Normal", delta_color="inverse" if alert_val > 0 else "off")

    st.markdown("---")

    # If no animal data recorded yet, display informative empty state
    if kpis["total_animals"] == 0:
        st.info(
            "No livestock monitoring data recorded yet. "
            "Navigate to **Live Monitoring** to upload a video or start webcam surveillance to begin analysis."
        )
        return

    # 2. Charts Section
    col_chart_left, col_chart_right = st.columns([1, 1])

    with col_chart_left:
        st.subheader("Cumulative Behavior Distribution")
        durations = db_manager.get_aggregated_behavior_durations()
        has_durations = any(v > 0 for v in durations.values())

        if has_durations:
            labels = list(durations.keys())
            values = list(durations.values())
            colors = [BEHAVIOR_COLORS_HEX.get(k, "#6c757d") for k in labels]

            fig_donut = go.Figure(
                data=[
                    go.Pie(
                        labels=labels,
                        values=values,
                        hole=0.48,
                        marker=dict(colors=colors),
                        textinfo="label+percent",
                        hoverinfo="label+value",
                    )
                ]
            )
            fig_donut.update_layout(
                margin=dict(l=15, r=15, t=25, b=15),
                height=300,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            )
            st.plotly_chart(fig_donut, use_container_width=True)
        else:
            st.caption("No completed behavior duration intervals logged yet.")

    with col_chart_right:
        st.subheader("Current Animal Behaviors")
        animals = db_manager.get_all_animals()
        if animals:
            df_animals = pd.DataFrame(animals)
            behavior_counts = df_animals["current_behavior"].value_counts().reset_index()
            behavior_counts.columns = ["Behavior", "Count"]

            bar_colors = [BEHAVIOR_COLORS_HEX.get(b, "#6c757d") for b in behavior_counts["Behavior"]]

            fig_bar = px.bar(
                behavior_counts,
                x="Behavior",
                y="Count",
                color="Behavior",
                color_discrete_map=BEHAVIOR_COLORS_HEX,
                text="Count",
            )
            fig_bar.update_layout(
                margin=dict(l=15, r=15, t=25, b=15),
                height=300,
                showlegend=False,
                xaxis_title="",
                yaxis_title="Count of Animals",
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    # 3. Recent Alerts Table
    st.subheader("Recent Inactivity Alerts")
    recent_alerts = db_manager.get_all_alerts(limit=5)
    if recent_alerts:
        df_alerts = pd.DataFrame(recent_alerts)
        df_display = df_alerts[["id", "tracking_id", "severity", "inactivity_seconds", "message", "created_at", "resolved"]].copy()
        df_display["inactivity_seconds"] = df_display["inactivity_seconds"].apply(seconds_to_hms)
        df_display["resolved"] = df_display["resolved"].apply(lambda r: "Resolved" if r == 1 else "Active")
        df_display.columns = ["Alert ID", "Cow #", "Severity", "Duration", "Message", "Timestamp (UTC)", "Status"]
        st.dataframe(df_display, use_container_width=True, hide_index=True)
    else:
        st.success("No inactivity alerts currently recorded. All animals operating within normal activity parameters.")
