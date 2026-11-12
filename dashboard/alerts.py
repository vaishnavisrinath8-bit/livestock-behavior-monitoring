"""
Inactivity Alerts Dashboard View
AI-Based Real-Time Livestock Behavior & Inactivity Monitoring System
"""

import streamlit as st
import pandas as pd
from database.database_operations import DatabaseManager
from utils.time_utils import seconds_to_hms


def render_alerts(db_manager: DatabaseManager) -> None:
    """Renders comprehensive alert registry with severity badges and manual/auto resolution states."""
    st.subheader("Inactivity & Welfare Alerts Registry")

    filter_choice = st.radio(
        "Filter Alerts",
        options=["All Alerts", "Unresolved Alerts Only", "Resolved Alerts Only"],
        horizontal=True,
    )

    if filter_choice == "Unresolved Alerts Only":
        resolved_filter = False
    elif filter_choice == "Resolved Alerts Only":
        resolved_filter = True
    else:
        resolved_filter = None

    alerts = db_manager.get_all_alerts(resolved_filter=resolved_filter, limit=200)

    if not alerts:
        st.success("No alerts found matching the selected filter.")
        return

    df = pd.DataFrame(alerts)
    df_display = pd.DataFrame()
    df_display["Alert ID"] = df["id"]
    df_display["Animal"] = df["tracking_id"].apply(lambda x: f"Cow #{x}")
    df_display["Alert Type"] = df["alert_type"]
    df_display["Severity"] = df["severity"]
    df_display["Inactivity Duration"] = df["inactivity_seconds"].apply(seconds_to_hms)
    df_display["Message"] = df["message"]
    df_display["Timestamp (UTC)"] = df["created_at"]
    df_display["Status"] = df["resolved"].apply(lambda r: "Resolved" if r == 1 else "Active")

    st.dataframe(df_display, use_container_width=True, hide_index=True)

    # Allow user to manually resolve open alerts if needed
    st.markdown("---")
    st.markdown("##### Alert Management")
    unresolved = [a for a in alerts if a["resolved"] == 0]
    if unresolved:
        unresolved_cows = sorted(list(set(a["tracking_id"] for a in unresolved)))
        col_sel, col_act = st.columns([2, 1])
        with col_sel:
            cow_to_resolve = st.selectbox("Resolve open alerts for Cow #", options=unresolved_cows)
        with col_act:
            st.write("")
            st.write("")
            if st.button("Mark Resolved", key="resolve_btn"):
                count = db_manager.resolve_alerts_for_animal(cow_to_resolve)
                st.success(f"Resolved {count} open alert(s) for Cow #{cow_to_resolve}.")
                st.rerun()
