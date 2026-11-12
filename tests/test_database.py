"""
Unit and Integration Tests for SQLite Database Module
AI-Based Real-Time Livestock Behavior & Inactivity Monitoring System
"""

import os
import pytest
import sqlite3
from pathlib import Path
from database.db import init_database, get_connection
from database.database_operations import DatabaseManager


@pytest.fixture
def test_db_path(tmp_path):
    """Provides an isolated temporary database for each test run."""
    db_file = tmp_path / "test_livestock.db"
    init_database(db_file)
    return db_file


@pytest.fixture
def db_mgr(test_db_path):
    """Provides a DatabaseManager bound to the test database."""
    return DatabaseManager(test_db_path)


def test_database_initialization(test_db_path):
    """Verifies that all required tables and indexes exist after init_database."""
    conn = get_connection(test_db_path)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = {row["name"] for row in cur.fetchall()}
    conn.close()

    assert "animals" in tables
    assert "activity_logs" in tables
    assert "alerts" in tables


def test_animal_upsert_and_retrieval(db_mgr):
    """Verifies inserting a new animal and subsequently updating last_seen and behavior."""
    # Insert new animal
    db_mgr.upsert_animal(
        tracking_id=1,
        timestamp="2026-11-12T10:00:00Z",
        current_behavior="Standing",
        total_observed_seconds=5.0,
        status="Active",
    )

    animal = db_mgr.get_animal(1)
    assert animal is not None
    assert animal["tracking_id"] == 1
    assert animal["current_behavior"] == "Standing"
    assert animal["total_observed_seconds"] == 5.0
    assert animal["status"] == "Active"

    # Upsert with updated behavior
    db_mgr.upsert_animal(
        tracking_id=1,
        timestamp="2026-11-12T10:01:00Z",
        current_behavior="Feeding",
        total_observed_seconds=15.0,
        status="Active",
    )

    updated = db_mgr.get_animal(1)
    assert updated["current_behavior"] == "Feeding"
    assert updated["total_observed_seconds"] == 15.0


def test_activity_log_insertion_and_aggregation(db_mgr):
    """Verifies inserting activity logs and computing duration summaries."""
    db_mgr.upsert_animal(1, "2026-11-12T10:00:00Z", "Standing")
    
    log_id1 = db_mgr.insert_activity_log(1, "Standing", 45.0, "2026-11-12T10:00:45Z")
    log_id2 = db_mgr.insert_activity_log(1, "Feeding", 30.0, "2026-11-12T10:01:15Z")

    assert log_id1 > 0
    assert log_id2 > log_id1

    logs = db_mgr.get_activity_logs(tracking_id=1)
    assert len(logs) == 2

    durations = db_mgr.get_behavior_duration_by_animal(1)
    assert durations["Standing"] == 45.0
    assert durations["Feeding"] == 30.0


def test_alert_lifecycle_and_resolution(db_mgr):
    """Verifies alert insertion, querying, and automatic resolution."""
    db_mgr.upsert_animal(2, "2026-11-12T10:00:00Z", "Lying/Resting")

    alert_id = db_mgr.insert_alert(
        tracking_id=2,
        alert_type="Prolonged Inactivity",
        inactivity_seconds=950.0,
        severity="Critical",
        message="Cow #2 prolonged lying observed.",
        created_at="2026-11-12T10:15:50Z",
    )
    assert alert_id > 0

    unresolved = db_mgr.get_unresolved_alert(2)
    assert unresolved is not None
    assert unresolved["severity"] == "Critical"
    assert unresolved["resolved"] == 0

    resolved_count = db_mgr.resolve_alerts_for_animal(2)
    assert resolved_count == 1

    post_check = db_mgr.get_unresolved_alert(2)
    assert post_check is None


def test_dashboard_kpis(db_mgr):
    """Verifies that KPI counts accurately calculate counts from actual database state."""
    db_mgr.upsert_animal(10, "2026-11-12T10:00:00Z", "Feeding", status="Active")
    db_mgr.upsert_animal(11, "2026-11-12T10:00:00Z", "Lying/Resting", status="Active")
    db_mgr.upsert_animal(12, "2026-11-12T10:00:00Z", "Standing", status="Lost")

    db_mgr.insert_alert(11, "Inactivity", 30.0, "Warning", "Lying long", "2026-11-12T10:00:30Z")

    kpis = db_mgr.get_dashboard_kpis()
    assert kpis["total_animals"] == 3
    assert kpis["active_animals"] == 2
    assert kpis["feeding_count"] == 1
    assert kpis["resting_count"] == 1
    assert kpis["active_alerts"] == 1
