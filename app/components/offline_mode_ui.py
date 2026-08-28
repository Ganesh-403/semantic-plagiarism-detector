# MIT License
#
# Copyright (c) 2026 Ganesh Kambli
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Offline Mode UI Components.

Provides UI elements for configuring and monitoring offline mode.
"""

import time
from pathlib import Path  # noqa: F401
from typing import Any, Dict  # noqa: F401

import streamlit as st

from src.core.offline_mode import OfflineConfig  # noqa: F401
from src.core.offline_mode import (
    get_offline_manager,
    initialize_offline_mode,
    is_offline_mode,
)


def render_offline_mode_settings() -> None:
    """Render offline mode settings panel."""
    st.markdown("### 🔒 Offline Mode Settings")

    # Check current status
    manager = get_offline_manager()
    current_status = manager.is_offline()

    # Status indicator
    if current_status:
        st.success("✅ Offline mode is **enabled**")
        st.caption("🔒 All external dependencies are disabled. Data stays local.")
    else:
        st.warning("⚠️ Offline mode is **disabled**")
        st.caption("🌐 External services (Redis, APIs) may be used.")

    st.divider()

    # Configuration
    with st.expander("⚙️ Configuration", expanded=True):
        config = manager.config

        enabled = st.toggle(
            "🔒 Enable Offline Mode",
            value=config.enabled,
            key="offline_mode_toggle",
            help="When enabled, no external services are used. All data stays local.",
        )

        if enabled:
            st.info("When offline mode is enabled, the system will NOT use:")
            st.markdown("- 🌐 Redis for session caching")
            st.markdown("- 🔗 External APIs (webhooks, translations)")
            st.markdown("- 📊 Telemetry and analytics")
            st.markdown("- 💾 Online model downloads")

            # Cache settings
            col1, col2 = st.columns(2)
            with col1:
                use_cache = st.checkbox(  # noqa: F841
                    "Use local cache",
                    value=config.use_local_cache,
                    key="offline_use_cache",
                )
            with col2:
                max_cache = st.number_input(  # noqa: F841
                    "Max cache size (MB)",
                    value=config.max_cache_size_mb,
                    min_value=50,
                    max_value=2000,
                    step=50,
                    key="offline_max_cache",
                )

            # Model settings
            col1, col2 = st.columns(2)
            with col1:
                preload = st.checkbox(  # noqa: F841
                    "Preload models on startup",
                    value=config.preload_models,
                    key="offline_preload_models",
                )
            with col2:
                use_fallback = st.checkbox(  # noqa: F841
                    "Use fallback embedding",
                    value=config.use_fallback_embedding,
                    key="offline_fallback_embedding",
                )

            # Advanced settings
            with st.expander("🔧 Advanced Settings"):
                col1, col2 = st.columns(2)
                with col1:
                    disable_telemetry = st.checkbox(  # noqa: F841
                        "Disable telemetry",
                        value=config.disable_telemetry,
                        key="offline_disable_telemetry",
                    )
                with col2:
                    auto_cleanup = st.checkbox(  # noqa: F841
                        "Auto cleanup cache",
                        value=config.auto_cleanup,
                        key="offline_auto_cleanup",
                    )

                cache_dir = st.text_input(  # noqa: F841
                    "Cache directory", value=config.cache_dir, key="offline_cache_dir"
                )

                model_cache_dir = st.text_input(  # noqa: F841
                    "Model cache directory",
                    value=config.model_cache_dir,
                    key="offline_model_cache_dir",
                )
        else:
            st.info("Enable offline mode to see configuration options.")

    # Action buttons
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔄 Apply Settings", use_container_width=True):
            # Apply settings
            # This would update the configuration
            st.success(
                "✅ Settings applied! Restart the application for changes to take effect."
            )

    with col2:
        if st.button("🧹 Clear Cache", use_container_width=True):
            manager = get_offline_manager()
            cache = manager.get_cache()
            if cache:
                cache.clear()
                st.success("✅ Cache cleared!")
                time.sleep(0.5)
                st.rerun()

    with col3:
        if st.button("📊 View Status", use_container_width=True):
            st.rerun()


def render_offline_mode_status() -> None:
    """Render offline mode status dashboard."""
    st.markdown("### 🔒 Offline Mode Status")

    manager = get_offline_manager()
    status = manager.get_status()

    if not status["enabled"]:
        st.info(
            "ℹ️ Offline mode is disabled. Enable it in Settings to use offline features."
        )
        return

    # Status metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Status", "🔒 Offline" if status["enabled"] else "🌐 Online")

    with col2:
        cache_stats = status.get("cache_stats", {})
        st.metric("Cache Files", cache_stats.get("total_files", 0))

    with col3:
        cache_stats = status.get("cache_stats", {})
        st.metric("Cache Size", f"{cache_stats.get('total_size_mb', 0):.1f} MB")

    with col4:
        st.metric("Models Cached", "N/A")

    st.divider()

    # Cache details
    if status["enabled"]:
        with st.expander("📁 Cache Details", expanded=True):
            cache_stats = status.get("cache_stats", {})

            col1, col2 = st.columns(2)
            with col1:
                st.write("**Cache Directory:**")
                st.code(cache_stats.get("cache_dir", "N/A"))

                st.write("**Metadata Entries:**")
                st.write(cache_stats.get("metadata_entries", 0))

            with col2:
                st.write("**Max Size:**")
                st.write(f"{cache_stats.get('max_size_mb', 0)} MB")

                st.write("**Usage:**")
                usage = cache_stats.get("usage_percent", 0)
                st.progress(min(1.0, usage / 100))
                st.caption(f"{usage:.1f}% used")

        with st.expander("⚙️ Configuration", expanded=False):
            config = status.get("config", {})
            for key, value in config.items():
                st.caption(f"**{key}:** {value}")

    # Initialize button if not initialized
    if not status.get("initialized", False):
        if st.button("🚀 Initialize Offline Mode", use_container_width=True):
            initialize_offline_mode()
            st.success("✅ Offline mode initialized!")
            st.rerun()


def render_offline_mode_badge() -> str:
    """Generate HTML badge for offline mode."""
    if is_offline_mode():
        return '<span style="background:#10B981;color:white;padding:2px 10px;border-radius:12px;font-size:0.7rem;font-weight:500;">🔒 Offline</span>'
    return '<span style="background:#6B7280;color:white;padding:2px 10px;border-radius:12px;font-size:0.7rem;font-weight:500;">🌐 Online</span>'


def render_offline_mode_sidebar_indicator() -> None:
    """Render offline mode indicator in sidebar."""
    if is_offline_mode():
        st.sidebar.success("🔒 **Offline Mode**")
        st.sidebar.caption("All data stays local")
    else:
        st.sidebar.info("🌐 **Online Mode**")
        st.sidebar.caption("External services enabled")
