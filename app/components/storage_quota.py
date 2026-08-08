"""
storage_quota.py
----------------
Settings component for rendering storage quota progress bar and usage caption.
"""

from pathlib import Path
from typing import Any, Dict
import streamlit as st
from src.utils.storage_metrics import calculate_storage_usage

STORAGE_LIMIT_GB = 10.0


def get_total_corpus_storage_bytes() -> int:
    """Calculate total byte size of stored corpus files, SQLite databases, and FAISS index files.

    Returns:
        Total bytes consumed by application storage assets.
    """
    usage = calculate_storage_usage()
    total_bytes = usage.get("total_bytes", 0)

    base_dir = Path(__file__).resolve().parents[2]
    data_dir = base_dir / "data"
    if data_dir.exists():
        for p in data_dir.glob("**/*"):
            if p.is_file() and p.suffix not in (".db", ".index"):
                try:
                    total_bytes += p.stat().st_size
                except OSError:
                    pass
    return total_bytes


def render_storage_quota_progress(limit_gb: float = 10.0) -> Dict[str, Any]:
    """Render st.progress bar in Settings tab displaying percentage of storage limit consumed.

    Args:
        limit_gb: Storage quota limit in Gigabytes (default 10.0 GB).

    Returns:
        Dict with storage statistics.
    """
    total_bytes = get_total_corpus_storage_bytes()
    total_gb = total_bytes / (1024**3)

    limit_bytes = int(limit_gb * 1024 * 1024 * 1024)
    percent = min(max(0.0, total_bytes / limit_bytes), 1.0)
    percent_display = int(percent * 100)

    st.markdown("### 💾 Storage Quota Gauge")
    st.progress(percent)
    caption_text = f"Storage Used: {total_gb:.1f} GB / {limit_gb:.1f} GB ({percent_display}%)"
    st.caption(caption_text)

    return {
        "total_bytes": total_bytes,
        "total_gb": total_gb,
        "limit_gb": limit_gb,
        "percent": percent,
        "caption": caption_text,
    }
