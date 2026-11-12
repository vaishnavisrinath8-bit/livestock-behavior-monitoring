"""
Database Connection and Schema Management
AI-Based Real-Time Livestock Behavior & Inactivity Monitoring System
"""

import sqlite3
import logging
from pathlib import Path
from typing import Optional
from config import DB_PATH

logger = logging.getLogger(__name__)


def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """
    Establish a thread-safe connection to the SQLite database.
    Enables foreign keys and returns rows as dictionaries/Row objects.
    """
    target_path = Path(db_path) if db_path else DB_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(str(target_path), check_same_thread=False, timeout=15.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")  # High concurrency & safety for dashboard reads
    return conn


def init_database(db_path: Optional[Path] = None) -> None:
    """
    Automatically creates required database tables and indexes if they do not exist.
    """
    conn = get_connection(db_path)
    try:
        with conn:
            # 1. Animals Table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS animals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tracking_id INTEGER UNIQUE NOT NULL,
                    first_seen TEXT NOT NULL,
                    last_seen TEXT NOT NULL,
                    current_behavior TEXT DEFAULT 'Unknown',
                    total_observed_seconds REAL DEFAULT 0.0,
                    status TEXT DEFAULT 'Active'
                );
            """)

            # 2. Activity Logs Table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS activity_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tracking_id INTEGER NOT NULL,
                    behavior TEXT NOT NULL,
                    duration_seconds REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    source_type TEXT DEFAULT 'video',
                    FOREIGN KEY (tracking_id) REFERENCES animals (tracking_id) ON DELETE CASCADE
                );
            """)

            # 3. Alerts Table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tracking_id INTEGER NOT NULL,
                    alert_type TEXT NOT NULL,
                    inactivity_seconds REAL NOT NULL,
                    severity TEXT NOT NULL CHECK(severity IN ('Normal', 'Warning', 'Critical')),
                    message TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    resolved INTEGER DEFAULT 0 CHECK(resolved IN (0, 1)),
                    FOREIGN KEY (tracking_id) REFERENCES animals (tracking_id) ON DELETE CASCADE
                );
            """)

            # Efficient Indexing
            conn.execute("CREATE INDEX IF NOT EXISTS idx_animals_tracking_id ON animals(tracking_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_activity_logs_tracking ON activity_logs(tracking_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_activity_logs_timestamp ON activity_logs(timestamp);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_tracking ON alerts(tracking_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_resolved ON alerts(resolved);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_created_at ON alerts(created_at);")

        logger.info("Database initialized successfully at %s", db_path or DB_PATH)
    except sqlite3.Error as e:
        logger.error("Failed to initialize database: %s", e)
        raise
    finally:
        conn.close()
