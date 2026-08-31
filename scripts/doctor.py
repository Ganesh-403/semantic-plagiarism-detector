#!/usr/bin/env python3
"""
scripts/doctor.py
-----------------
Administrative diagnostic utility for pre-flight environment checks.
Validates Python version, SQLite WAL mode, Tesseract, Redis, FAISS,
and disk permissions.

Usage:
    python scripts/doctor.py

Exit Codes:
    0 - All checks PASS or WARN.
    1 - One or more checks FAIL.
"""

from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

import psutil

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT_DIR / ".env")
except ImportError:
    pass

APP_CONFIG_LOADED = True
try:
    from src.core.app_config import CORPUS_DB_PATH, DATA_DIR, LOGS_DIR
except Exception:
    APP_CONFIG_LOADED = False
    DATA_DIR = ROOT_DIR / "data"
    LOGS_DIR = ROOT_DIR / "logs"
    CORPUS_DB_PATH = DATA_DIR / "corpus.db"

# ── Colors ───────────────────────────────────────────────────────────────────

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"


def print_status(name: str, status: str, message: str) -> None:
    """Print a color-coded status line."""
    if status == "PASS":
        color = GREEN
    elif status == "WARN":
        color = YELLOW
    else:
        color = RED

    print(f"[{color}{status}{RESET}] {name:<20} : {message}")


# ── Checks ───────────────────────────────────────────────────────────────────


def check_python_version() -> str:
    """Check if Python version is >= 3.10."""
    if sys.version_info >= (3, 10):
        print_status("Python Version", "PASS", f"{sys.version.split()[0]} detected")
        return "PASS"
    else:
        print_status(
            "Python Version",
            "FAIL",
            f"{sys.version.split()[0]} detected, require 3.10+",
        )
        return "FAIL"


def check_sqlite_wal() -> str:
    """Check if SQLite DB exists and is in WAL mode."""
    if not CORPUS_DB_PATH.exists():
        print_status(
            "SQLite Database", "WARN", f"Database not found at {CORPUS_DB_PATH}"
        )
        return "WARN"

    try:
        from contextlib import closing

        with closing(sqlite3.connect(CORPUS_DB_PATH)) as conn:
            cursor = conn.execute("PRAGMA journal_mode")
            row = cursor.fetchone()
            if row:
                mode = row[0].lower()
                if mode == "wal":
                    print_status("SQLite WAL Mode", "PASS", "WAL mode is enabled")
                    return "PASS"
                else:
                    print_status(
                        "SQLite WAL Mode",
                        "FAIL",
                        f"Current mode is '{mode}', expected 'wal'",
                    )
                    return "FAIL"
            else:
                print_status("SQLite WAL Mode", "FAIL", "Failed to fetch journal_mode")
                return "FAIL"
    except Exception as e:
        print_status("SQLite Database", "FAIL", f"Error connecting: {e}")
        return "FAIL"


def check_tesseract() -> str:
    """Check if Tesseract is installed and available."""
    try:
        import pytesseract

        version = pytesseract.get_tesseract_version()
        print_status("Tesseract", "PASS", f"Version {version} detected")
        return "PASS"
    except ImportError:
        print_status("Tesseract", "FAIL", "pytesseract not installed")
        return "FAIL"
    except Exception:
        print_status(
            "Tesseract",
            "FAIL",
            "Binary not found. Please install Tesseract "
            "(e.g., 'sudo apt install tesseract-ocr' or via Homebrew/Windows installer).",
        )
        return "FAIL"


def check_redis() -> str:
    """Check Redis connectivity."""
    try:
        import redis
    except ImportError:
        print_status("Redis", "FAIL", "redis module not installed")
        return "FAIL"

    redis_url = os.environ.get("REDIS_URL")
    if not redis_url:
        print_status("Redis", "WARN", "REDIS_URL not configured in environment")
        return "WARN"

    try:
        client = redis.Redis.from_url(redis_url, socket_timeout=2)
        if client.ping():
            print_status("Redis", "PASS", "Successfully connected and pinged")
            return "PASS"
        else:
            print_status("Redis", "FAIL", "Ping returned false")
            return "FAIL"
    except Exception as e:
        print_status("Redis", "FAIL", f"Connection failed: {e}")
        return "FAIL"


def check_faiss() -> str:
    """Check if FAISS loads natively."""
    try:
        import faiss
    except ImportError:
        print_status("FAISS", "FAIL", "faiss module not importable")
        return "FAIL"
    except Exception as e:
        print_status("FAISS", "FAIL", f"Native library load error: {e}")
        return "FAIL"

    try:
        index = faiss.IndexFlatL2(1)
        if index is not None:
            print_status("FAISS", "PASS", "Native library loaded and operational")
            return "PASS"
        else:
            print_status("FAISS", "FAIL", "Failed to create test index")
            return "FAIL"
    except Exception as e:
        print_status("FAISS", "FAIL", f"Error creating test index: {e}")
        return "FAIL"


def check_disk() -> str:
    """Check disk permissions and space."""
    overall_status = "PASS"

    # Permissions check
    for d in [DATA_DIR, LOGS_DIR]:
        if not d.exists():
            try:
                d.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                print_status("Disk Permissions", "FAIL", f"Cannot create {d}: {e}")
                overall_status = "FAIL"
                continue

        if not os.access(d, os.W_OK):
            print_status("Disk Permissions", "FAIL", f"No write access to {d}")
            overall_status = "FAIL"

    if overall_status == "FAIL":
        return "FAIL"

    # Space check (warn if < 1GB)
    try:
        usage = psutil.disk_usage(str(DATA_DIR))
        free_gb = usage.free / (1024**3)
        if free_gb < 1.0:
            print_status(
                "Disk Space", "WARN", f"Only {free_gb:.2f} GB free on {DATA_DIR}"
            )
            overall_status = "WARN" if overall_status == "PASS" else overall_status
        else:
            print_status(
                "Disk Permissions", "PASS", f"Write access OK, {free_gb:.2f} GB free"
            )
    except Exception as e:
        print_status("Disk Space", "WARN", f"Could not determine free space: {e}")
        overall_status = "WARN" if overall_status == "PASS" else overall_status

    return overall_status


def main() -> int:
    """Run all checks and return exit code."""
    # Enable VT100 ANSI escape sequence processing in older Windows consoles
    if os.name == "nt":
        os.system("")

    print("========================================")
    print(" Pre-flight Environment Check")
    print("========================================")
    print()

    if not APP_CONFIG_LOADED:
        print_status(
            "App Config",
            "WARN",
            "Could not import src.core.app_config (missing dependencies?). Using fallback paths.",
        )

    results = [
        "WARN" if not APP_CONFIG_LOADED else "PASS",
        check_python_version(),
        check_sqlite_wal(),
        check_tesseract(),
        check_redis(),
        check_faiss(),
        check_disk(),
    ]

    print()
    print("-" * 40)
    if "FAIL" in results:
        print(f"[{RED}FAILED{RESET}] Environment has critical errors.")
        return 1
    elif "WARN" in results:
        print(f"[{YELLOW}WARNING{RESET}] Environment is ready with warnings.")
        return 0
    else:
        print(f"[{GREEN}SUCCESS{RESET}] Environment is fully ready.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
