"""Administrator system health panel."""

from pathlib import Path

import psutil
import streamlit as st

from src.db.corpus_db import get_corpus_db_path
from src.utils.redis_cache import get_cache


def format_storage_size(size_bytes: int) -> str:
    if size_bytes >= 1_048_576:
        return f"{size_bytes / 1_048_576:.2f} MB"
    if size_bytes >= 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes} B"


def collect_system_health():
    memory = psutil.virtual_memory()
    path = Path(get_corpus_db_path())
    try:
        connected, latency = get_cache().ping()
    except Exception:
        connected, latency = False, 0
    return {
        "cpu": psutil.cpu_percent(interval=None),
        "memory_percent": memory.percent,
        "memory_used": memory.used,
        "redis_connected": connected,
        "redis_latency": latency,
        "database_bytes": path.stat().st_size if path.is_file() else 0,
    }


def render_health_view(user_role: str):
    if user_role != "admin":
        st.info("System health is available to administrators.")
        return
    st.subheader("System Health")
    if st.button("Refresh Metrics", key="health_refresh_button"):
        st.rerun()
    try:
        health = collect_system_health()
    except (OSError, RuntimeError) as exc:
        st.error(f"System health is unavailable: {exc}")
        return
    cpu, memory, database = st.columns(3)
    cpu.metric("CPU Usage", f"{health['cpu']:.1f}%")
    memory.metric("Memory Usage", f"{health['memory_percent']:.1f}%")
    memory.caption(f"Memory Used: {format_storage_size(health['memory_used'])}")
    database.metric("Database Size", format_storage_size(health["database_bytes"]))
    st.metric("Redis", "Connected" if health["redis_connected"] else "Disconnected")
    if health["redis_connected"]:
        st.caption(f"Redis latency: {health['redis_latency']:.1f} ms")
    else:
        st.warning("Redis is unavailable; the application will use its local fallback.")
