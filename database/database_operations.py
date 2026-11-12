"""
Database Operations Interface
AI-Based Real-Time Livestock Behavior & Inactivity Monitoring System
"""

import sqlite3
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from database.db import get_connection

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages transactional CRUD operations for animals, logs, and alerts."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path

    def _get_conn(self) -> sqlite3.Connection:
        return get_connection(self.db_path)

    # ------------------ ANIMALS ------------------

    def upsert_animal(
        self,
        tracking_id: int,
        timestamp: str,
        current_behavior: str = "Unknown",
        total_observed_seconds: float = 0.0,
        status: str = "Active",
    ) -> None:
        """
        Inserts a newly tracked animal or updates the existing animal's
        last_seen, current_behavior, total_observed_seconds, and status.
        """
        sql = """
            INSERT INTO animals (
                tracking_id, first_seen, last_seen, current_behavior,
                total_observed_seconds, status
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(tracking_id) DO UPDATE SET
                last_seen = excluded.last_seen,
                current_behavior = excluded.current_behavior,
                total_observed_seconds = excluded.total_observed_seconds,
                status = excluded.status;
        """
        conn = self._get_conn()
        try:
            with conn:
                conn.execute(
                    sql,
                    (
                        tracking_id,
                        timestamp,
                        timestamp,
                        current_behavior,
                        total_observed_seconds,
                        status,
                    ),
                )
        except sqlite3.Error as e:
            logger.error("Error upserting animal #%s: %s", tracking_id, e)
            raise
        finally:
            conn.close()

    def get_animal(self, tracking_id: int) -> Optional[Dict[str, Any]]:
        """Retrieves an animal record by tracking ID."""
        sql = "SELECT * FROM animals WHERE tracking_id = ?;"
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute(sql, (tracking_id,))
            row = cur.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_all_animals(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieves all tracked animals, optionally filtered by status."""
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            if status:
                cur.execute(
                    "SELECT * FROM animals WHERE status = ? ORDER BY tracking_id ASC;",
                    (status,),
                )
            else:
                cur.execute("SELECT * FROM animals ORDER BY tracking_id ASC;")
            return [dict(row) for row in cur.fetchall()]
        finally:
            conn.close()

    def update_animal_status(self, tracking_id: int, status: str) -> None:
        """Updates the status of an animal (e.g. 'Active', 'Lost', 'Archived')."""
        sql = "UPDATE animals SET status = ? WHERE tracking_id = ?;"
        conn = self._get_conn()
        try:
            with conn:
                conn.execute(sql, (status, tracking_id))
        finally:
            conn.close()

    # ------------------ ACTIVITY LOGS ------------------

    def insert_activity_log(
        self,
        tracking_id: int,
        behavior: str,
        duration_seconds: float,
        timestamp: str,
        source_type: str = "video",
    ) -> int:
        """Logs a completed behavior duration event for a tracked animal."""
        sql = """
            INSERT INTO activity_logs (
                tracking_id, behavior, duration_seconds, timestamp, source_type
            ) VALUES (?, ?, ?, ?, ?);
        """
        conn = self._get_conn()
        try:
            with conn:
                cur = conn.cursor()
                cur.execute(
                    sql,
                    (
                        tracking_id,
                        behavior,
                        round(duration_seconds, 2),
                        timestamp,
                        source_type,
                    ),
                )
                return cur.lastrowid
        except sqlite3.Error as e:
            logger.error("Error inserting activity log: %s", e)
            raise
        finally:
            conn.close()

    def get_activity_logs(
        self,
        tracking_id: Optional[int] = None,
        limit: int = 100,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieves activity logs with optional filtering by tracking ID and date range."""
        query = "SELECT * FROM activity_logs WHERE 1=1"
        params: List[Any] = []

        if tracking_id is not None:
            query += " AND tracking_id = ?"
            params.append(tracking_id)
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)

        query += " ORDER BY id DESC LIMIT ?;"
        params.append(limit)

        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute(query, tuple(params))
            return [dict(row) for row in cur.fetchall()]
        finally:
            conn.close()

    def get_behavior_duration_by_animal(self, tracking_id: int) -> Dict[str, float]:
        """Calculates total historical duration spent in each behavior for an animal."""
        sql = """
            SELECT behavior, SUM(duration_seconds) as total_duration
            FROM activity_logs
            WHERE tracking_id = ?
            GROUP BY behavior;
        """
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute(sql, (tracking_id,))
            durations = {row["behavior"]: float(row["total_duration"]) for row in cur.fetchall()}
            return durations
        finally:
            conn.close()

    def get_aggregated_behavior_durations(self) -> Dict[str, float]:
        """Calculates total cumulative duration for each behavior across all animals."""
        sql = """
            SELECT behavior, SUM(duration_seconds) as total_duration
            FROM activity_logs
            GROUP BY behavior;
        """
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute(sql)
            return {row["behavior"]: float(row["total_duration"] or 0.0) for row in cur.fetchall()}
        finally:
            conn.close()

    # ------------------ ALERTS ------------------

    def insert_alert(
        self,
        tracking_id: int,
        alert_type: str,
        inactivity_seconds: float,
        severity: str,
        message: str,
        created_at: str,
    ) -> int:
        """Records an inactivity alert."""
        sql = """
            INSERT INTO alerts (
                tracking_id, alert_type, inactivity_seconds, severity,
                message, created_at, resolved
            ) VALUES (?, ?, ?, ?, ?, ?, 0);
        """
        conn = self._get_conn()
        try:
            with conn:
                cur = conn.cursor()
                cur.execute(
                    sql,
                    (
                        tracking_id,
                        alert_type,
                        round(inactivity_seconds, 2),
                        severity,
                        message,
                        created_at,
                    ),
                )
                return cur.lastrowid
        except sqlite3.Error as e:
            logger.error("Error creating alert: %s", e)
            raise
        finally:
            conn.close()

    def get_unresolved_alert(self, tracking_id: int) -> Optional[Dict[str, Any]]:
        """Returns the current unresolved alert for an animal if one exists."""
        sql = """
            SELECT * FROM alerts
            WHERE tracking_id = ? AND resolved = 0
            ORDER BY id DESC LIMIT 1;
        """
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute(sql, (tracking_id,))
            row = cur.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def resolve_alerts_for_animal(self, tracking_id: int) -> int:
        """Marks any open alerts for a given animal as resolved."""
        sql = "UPDATE alerts SET resolved = 1 WHERE tracking_id = ? AND resolved = 0;"
        conn = self._get_conn()
        try:
            with conn:
                cur = conn.cursor()
                cur.execute(sql, (tracking_id,))
                return cur.rowcount
        finally:
            conn.close()

    def get_all_alerts(
        self,
        resolved_filter: Optional[bool] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Fetches all alerts with optional unresolved filter."""
        query = "SELECT * FROM alerts WHERE 1=1"
        params: List[Any] = []
        if resolved_filter is not None:
            query += " AND resolved = ?"
            params.append(1 if resolved_filter else 0)
        query += " ORDER BY id DESC LIMIT ?;"
        params.append(limit)

        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute(query, tuple(params))
            return [dict(row) for row in cur.fetchall()]
        finally:
            conn.close()

    # ------------------ KPI QUERIES ------------------

    def get_dashboard_kpis(self) -> Dict[str, Any]:
        """Calculates real-time metrics strictly from recorded database data."""
        conn = self._get_conn()
        try:
            cur = conn.cursor()

            # Total animals
            cur.execute("SELECT COUNT(*) as cnt FROM animals;")
            total_animals = cur.fetchone()["cnt"]

            # Active animals
            cur.execute("SELECT COUNT(*) as cnt FROM animals WHERE status = 'Active';")
            active_animals = cur.fetchone()["cnt"]

            # Feeding animals (currently)
            cur.execute("SELECT COUNT(*) as cnt FROM animals WHERE current_behavior = 'Feeding' AND status = 'Active';")
            feeding_count = cur.fetchone()["cnt"]

            # Resting / Lying animals (currently)
            cur.execute("SELECT COUNT(*) as cnt FROM animals WHERE current_behavior = 'Lying/Resting' AND status = 'Active';")
            resting_count = cur.fetchone()["cnt"]

            # Unresolved alerts
            cur.execute("SELECT COUNT(*) as cnt FROM alerts WHERE resolved = 0;")
            active_alerts = cur.fetchone()["cnt"]

            return {
                "total_animals": total_animals,
                "active_animals": active_animals,
                "feeding_count": feeding_count,
                "resting_count": resting_count,
                "active_alerts": active_alerts,
            }
        finally:
            conn.close()

    def clear_database(self) -> None:
        """Safely resets tables for fresh demonstration runs."""
        conn = self._get_conn()
        try:
            with conn:
                conn.execute("DELETE FROM alerts;")
                conn.execute("DELETE FROM activity_logs;")
                conn.execute("DELETE FROM animals;")
        finally:
            conn.close()
