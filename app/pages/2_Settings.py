"""
app/pages/2_Settings.py
-----------------------
Streamlit multi-page app: System Configuration and Settings.

This page allows administrators to configure plagiarism detection thresholds,
OCR settings, database management, and system backups.

Issue #2810: Decompose monolithic streamlit_app.py.
"""

import json
from pathlib import Path

import streamlit as st

from src.core.config import PLAGIARISM_THRESHOLD
from src.db.database_backup import (
    create_corpus_database_snapshot,
    create_password_protected_backup,
)

st.set_page_config(
    page_title="Settings - Plagiarism Detector", page_icon="⚙️", layout="wide"
)


def reset_settings():
    """Reset settings before Streamlit recreates the widgets on the next run."""
    defaults = {
        "threshold_slider": PLAGIARISM_THRESHOLD,
        "chunk_matrix_checkbox": False,
        "faiss_top_k_slider": 5,
        "settings_lexical_slider": 0.5,
        "settings_semantic_slider": 0.65,
        "ocr_language_selector": "English",
        "ocr_dpi_slider": 250,
    }
    for key, value in defaults.items():
        st.session_state[key] = value
    st.query_params.pop("threshold", None)


def render_settings():
    """Render the system settings and configuration UI."""
    st.title("⚙️ System Configuration")

    if not st.session_state.get("authenticated") or not st.session_state.get("username"):
        st.info("Sign in to manage your settings.")
        return
    from app.components.notification_preferences import render_notification_preferences

    render_notification_preferences()
    user_role = st.session_state.get("role", "user")
    if user_role != "admin":
        st.error(
            "🔒 Access Denied: Administrator privileges required to view settings."
        )
        return

    st.markdown("Configure detection thresholds, OCR parameters, and system backups.")

    tab_thresholds, tab_ocr, tab_backup = st.tabs(
        ["🎯 Thresholds", "🔤 OCR Settings", "💾 Backup & Export"]
    )

    with tab_thresholds:
        st.subheader("Plagiarism Detection Thresholds")

        threshold = st.slider(
            "Hybrid Similarity Threshold",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.get("threshold_slider", PLAGIARISM_THRESHOLD),
            step=0.01,
            key="threshold_slider",
            help="Combined Hybrid score threshold for flagging pair plagiarism.",
        )

        lexical_threshold = st.slider(
            "Lexical Sensitivity Threshold",
            0.0,
            1.0,
            value=0.50,
            step=0.01,
            help="Direct word-for-word and N-gram match threshold.",
            key="settings_lexical_slider",
        )

        semantic_threshold = st.slider(
            "Semantic Sensitivity Threshold",
            0.0,
            1.0,
            value=0.65,
            step=0.01,
            help="Transformer embedding vector similarity threshold.",
            key="settings_semantic_slider",
        )

        if st.button("💾 Save Thresholds", type="primary"):
            st.success("✅ Thresholds saved to session state.")
        st.checkbox("Use chunk similarity matrix", key="chunk_matrix_checkbox")
        st.slider("FAISS search results", 1, 20, 5, key="faiss_top_k_slider")
        st.button("🔄 Reset to Factory Defaults", key="reset_defaults_button", on_click=reset_settings)

    with tab_ocr:
        st.subheader("Optical Character Recognition (OCR)")
        st.caption("Used only for scanned or image-only PDF pages.")

        from src.core.document_parser import DEFAULT_OCR_DPI, SUPPORTED_OCR_LANGUAGES

        ocr_language_labels = {
            display_name: code for code, display_name in SUPPORTED_OCR_LANGUAGES.items()
        }

        selected_lang = st.selectbox(
            "OCR Language", options=list(ocr_language_labels.keys()), index=0,
            key="ocr_language_selector",
        )

        ocr_dpi = st.slider(
            "OCR DPI Resolution",
            min_value=150,
            max_value=400,
            value=DEFAULT_OCR_DPI,
            step=25,
            key="ocr_dpi_slider",
        )

    with tab_backup:
        st.subheader("Database Backup & Configuration Export")

        backup_password = st.text_input(
            "🔑 Backup Password (optional)",
            type="password",
            help="If set, the backup file will be AES-256-encrypted.",
        )

        if st.button("⬇️ Generate Database Backup"):
            with st.spinner("Creating snapshot..."):
                snapshot = create_corpus_database_snapshot()
                if backup_password:
                    backup_data = create_password_protected_backup(
                        snapshot, backup_password
                    )
                    st.download_button(
                        "⬇️ Download Encrypted Backup",
                        data=backup_data,
                        file_name="corpus_backup_encrypted.zip",
                        mime="application/zip",
                        key="download_raw_corpus_database",
                    )
                else:
                    st.download_button(
                        "⬇️ Download Raw Database",
                        data=snapshot,
                        file_name="corpus.db",
                        mime="application/vnd.sqlite3",
                        key="download_raw_corpus_database",
                    )

        st.divider()

        st.markdown("### 📥 Export Configuration (JSON)")
        config_data = {
            "threshold": st.session_state.get("threshold_slider", 0.59),
            "lexical_threshold": lexical_threshold,
            "semantic_threshold": semantic_threshold,
            "ocr_language": selected_lang,
            "ocr_dpi": ocr_dpi,
        }

        st.download_button(
            "📥 Download config.json",
            data=json.dumps(config_data, indent=2),
            file_name="plagiarism_config_backup.json",
            mime="application/json",
            key="backup_config_button",
        )

        if st.button("Check Database Schema", key="check_db_schema_btn"):
            from src.db import auth, corpus_db
            from src.db.connection import get_connection
            from src.db.migrations.common import get_user_version

            versions = []
            for label, path in (("Corpus", corpus_db._DB_PATH), ("Auth", auth._DB_PATH)):
                if Path(path).is_file():
                    with get_connection(path, read_only=True) as conn:
                        versions.append(f"{label} Schema: v{get_user_version(conn)}")
                else:
                    versions.append(f"{label} Schema: database not created")
            st.session_state["db_schema_status_msg"] = " | ".join(versions)
        if "db_schema_status_msg" in st.session_state:
            st.info(st.session_state["db_schema_status_msg"])


if __name__ == "__main__":
    render_settings()
