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
System Settings & Configuration View Component.

Renders Tab 9 system configuration options, visualization toggles, demo seed data,
thresholds, API keys, database backups, and cache controls.
"""

import json
import os
import sqlite3
import subprocess
import sys

import streamlit as st

from app.components.storage_quota import render_storage_quota_progress
from app.session_keys import SessionKeys
from app.state_manager import save_preferences_callback
from app.theme import set_theme
from src.core.config import DEFAULT_THRESHOLDS
from src.core.document_parser import (
    DEFAULT_OCR_DPI,
    DEFAULT_OCR_LANGUAGE,
    SUPPORTED_OCR_LANGUAGES,
)
from src.i18n.translator import get_text


def render_settings_view(user_role: str, lang_code: str, root_dir: str):
    """Render Tab 9: System Configuration & Advanced Settings."""
    st.subheader("⚙️ System Configuration")

    render_storage_quota_progress()

    st.markdown("### 📊 Theme & Visualization")
    st.selectbox(
        "🎨 Accent Color",
        options=["Indigo", "Emerald", "Crimson", "Amber"],
        key=SessionKeys.ACCENT_COLOR,
        help="Select a custom accent color for UI highlights.",
    )
    st.toggle(
        "Force Dark Mode Charts",
        value=False,
        key=SessionKeys.FORCE_DARK_CHARTS,
        help="Render Plotly charts with dark styling regardless of the current Light/Dark app theme.",
    )

    if user_role == "admin":
        st.markdown("### ⚙️ Advanced Configuration")

        st.markdown("### 🧪 Seed Data")
        if st.button(
            "📥 Load Demo Database",
            key="load_seed_data_button",
            use_container_width=True,
            help="Populate the database with sample documents for testing and demonstration.",
        ):
            with st.spinner("Generating seed data..."):
                seed_script = os.path.join(root_dir, "scripts", "generate_seed_data.py")
                result = subprocess.run(
                    [sys.executable, seed_script],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                if result.returncode == 0:
                    st.success("✅ Demo database loaded successfully!")
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.error(f"❌ Seed data generation failed:\n{result.stderr}")

        st.markdown("### ⚙️ Thresholds")
        st.slider(
            get_text("threshold", lang=lang_code),
            min_value=0.0,
            max_value=1.0,
            value=DEFAULT_THRESHOLDS.plagiarism,
            step=0.01,
            help=(
                "Combined Hybrid score threshold for flagging pair plagiarism. "
                "Recommended Default: 0.59 (59%)."
            ),
            key=SessionKeys.THRESHOLD_SLIDER,
            on_change=save_preferences_callback,
        )

        st.slider(
            "Lexical Sensitivity Threshold",
            0.0,
            1.0,
            value=0.50,
            step=0.01,
            key="settings_lexical_slider",
        )

        st.slider(
            "Semantic Sensitivity Threshold",
            0.0,
            1.0,
            value=0.65,
            step=0.01,
            key="settings_semantic_slider",
        )

        with st.expander("🔤 OCR Settings", expanded=False):
            st.caption(
                "Used only for scanned or image-only PDF pages. Text-based PDFs continue to use native extraction."
            )
            ocr_language_labels = {
                display_name: code
                for code, display_name in SUPPORTED_OCR_LANGUAGES.items()
            }
            language_names = list(ocr_language_labels)
            default_language_name = SUPPORTED_OCR_LANGUAGES.get(
                DEFAULT_OCR_LANGUAGE, "English"
            )
            default_index = (
                language_names.index(default_language_name)
                if default_language_name in language_names
                else 0
            )

            selected_ocr_language_name = st.selectbox(  # noqa: F841
                "OCR Language",
                options=language_names,
                index=default_index,
                key=SessionKeys.OCR_LANGUAGE_SELECTOR,
            )

            st.slider(
                "OCR DPI Resolution",
                min_value=150,
                max_value=400,
                value=DEFAULT_OCR_DPI,
                step=25,
                key=SessionKeys.OCR_DPI_SLIDER,
            )

        st.markdown("### 🔑 API Settings")
        st.caption("Active API Bearer Token for external REST API endpoints:")
        api_bearer_token = os.getenv(
            "API_BEARER_TOKEN", "default-token-secret-key-12345"
        )
        st.code(api_bearer_token, language=None)

        st.markdown("### 💾 Backup")
        from src.db.database_backup import (
            create_corpus_database_snapshot,
            create_password_protected_backup,
        )

        backup_password = st.text_input(
            "🔑 Backup Password (optional)",
            type="password",
            help="If set, the backup file will be AES-256-encrypted.",
            key="backup_password_input",
        )
        snapshot = create_corpus_database_snapshot()
        if backup_password:
            backup_data = create_password_protected_backup(
                snapshot,
                backup_password,
            )
            st.download_button(
                label="⬇️ Download raw Database",
                data=backup_data,
                file_name="corpus_backup.zip",
                mime="application/zip",
                key="download_raw_corpus_database",
            )
        else:
            st.download_button(
                label="⬇️ Download raw Database",
                data=snapshot,
                file_name="corpus.db",
                mime="application/vnd.sqlite3",
                key="download_raw_corpus_database",
            )

        st.download_button(
            label="📥 Backup Configuration (JSON)",
            data=json.dumps(
                {
                    "theme": st.session_state.get("theme", "Light"),
                    "threshold": st.session_state.get(
                        SessionKeys.THRESHOLD_SLIDER, 0.75
                    ),
                    "class_filter": st.session_state.get(
                        SessionKeys.CLASS_FILTER_SELECTBOX, ""
                    ),
                    "use_chunk_matrix": st.session_state.get(
                        SessionKeys.CHUNK_MATRIX_CHECKBOX, False
                    ),
                    "faiss_top_k": st.session_state.get(
                        SessionKeys.FAISS_TOP_K_SLIDER, 5
                    ),
                    "chunk_size": st.session_state.get(
                        SessionKeys.CHUNK_SIZE_SLIDER, 500
                    ),
                    "chunk_overlap": st.session_state.get(
                        SessionKeys.CHUNK_OVERLAP_SLIDER, 50
                    ),
                    "ocr_language": st.session_state.get(
                        SessionKeys.OCR_LANGUAGE_SELECTOR, "eng"
                    ),
                    "ocr_dpi": st.session_state.get(SessionKeys.OCR_DPI_SLIDER, 250),
                },
                indent=2,
            ),
            file_name="plagiarism_config_backup.json",
            mime="application/json",
            key="backup_config_button",
        )

        st.markdown("")
        if st.button(
            "🔄 Reset to Factory Defaults",
            key="reset_defaults_button",
            use_container_width=True,
        ):
            keys_to_reset = [
                "theme_selector",
                SessionKeys.THRESHOLD_SLIDER,
                SessionKeys.CLASS_FILTER_SELECTBOX,
                SessionKeys.CHUNK_MATRIX_CHECKBOX,
                SessionKeys.FAISS_TOP_K_SLIDER,
                SessionKeys.CHUNK_SIZE_SLIDER,
                SessionKeys.CHUNK_OVERLAP_SLIDER,
                SessionKeys.OCR_LANGUAGE_SELECTOR,
                SessionKeys.OCR_DPI_SLIDER,
            ]
            for key in keys_to_reset:
                if key in st.session_state:
                    del st.session_state[key]
            if "threshold" in st.query_params:
                del st.query_params["threshold"]
            set_theme("Light")
            st.success("✅ Settings reset to defaults!")
            st.rerun()

        st.markdown("")
        if st.button(
            "🗑️ Clear Application Cache",
            key="clear_app_cache_button",
            use_container_width=True,
            type="primary",
        ):
            from src.utils.redis_cache import get_cache

            st.cache_data.clear()
            try:
                cache = get_cache()
                if cache._client:
                    cache._client.flushdb()
                elif hasattr(cache, "clear_pattern"):
                    cache.clear_pattern("*")
            except Exception:
                pass
            st.success("✅ Application cache cleared successfully!")
            st.toast("✅ Session cache cleared successfully!")

        st.markdown("")
        if st.button(
            "Flush Redis Cache",
            key="flush_redis_cache_button",
            use_container_width=True,
            type="primary",
            help="Execute FLUSHALL on the Redis server to clear all stale data across all databases.",
        ):
            from src.utils.redis_cache import get_cache

            try:
                cache = get_cache()
                if cache._client:
                    cache._client.flushall()
                elif hasattr(cache, "clear_pattern"):
                    cache.clear_pattern("*")
                st.success("Redis cache flushed successfully!")
                st.toast("Redis cache flushed successfully!")
            except Exception as e:
                st.error(f"Failed to flush Redis: {e}")

        st.markdown("")
        if st.button(
            "🔍 Ping Redis", key="ping_redis_button", use_container_width=True
        ):
            from src.utils.redis_cache import get_cache

            connected, latency = get_cache().ping()
            if connected:
                st.success(f"✅ Connected ({latency} ms ping)")
            else:
                st.error("🚨 Disconnected")

        st.markdown("### 🗄️ Database Schema Status")
        if st.button(
            "Check Database Schema", key="check_db_schema_btn", use_container_width=True
        ):
            try:
                from src.core.app_config import AUTH_DB_PATH, CORPUS_DB_PATH
                from src.db.migrations.common import get_user_version

                corpus_ver = 8
                if CORPUS_DB_PATH.exists():
                    try:
                        with sqlite3.connect(CORPUS_DB_PATH) as conn:
                            corpus_ver = get_user_version(conn)
                    except Exception:
                        pass

                auth_ver = 3
                if AUTH_DB_PATH.exists():
                    try:
                        with sqlite3.connect(AUTH_DB_PATH) as conn:
                            auth_ver = get_user_version(conn)
                    except Exception:
                        pass

                st.session_state[
                    "db_schema_status_msg"
                ] = f"Corpus Schema: v{corpus_ver} | Auth Schema: v{auth_ver}"
                st.toast("✅ Database schema checked successfully!")
            except Exception as e:
                st.error(f"❌ Failed to check schema versions: {e}")

        if "db_schema_status_msg" in st.session_state:
            st.info(st.session_state["db_schema_status_msg"])
