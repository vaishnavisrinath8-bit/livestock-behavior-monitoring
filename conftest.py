"""
Pytest Global Configuration and Fixtures
AI-Based Real-Time Livestock Behavior & Inactivity Monitoring System
"""

import sys
from pathlib import Path

# Add project root to sys.path so modules can be imported directly
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
