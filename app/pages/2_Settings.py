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
app/pages/2_Settings.py
-----------------------
Streamlit multi-page app: System Configuration and Settings.

This page allows administrators to configure plagiarism detection thresholds,
OCR settings, database management, and system backups.

Issue #2810: Decompose monolithic streamlit_app.py.
"""

import json

import streamlit as st

from src.core.config import PLAGIARISM_THRESHOLD
from src.db.database_backup import (
    create_corpus_database_snapshot,
    create_password_protected_backup,
)

st.set_page_config(
    page_title="Settings - Plagiarism Detector", page_icon="⚙️", layout="wide"
)


def render_settings():
    """Render the system settings and configuration UI."""
    st.title("⚙️ System Configuration")

    # Check admin privileges (simplified for page context)
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
            help="Combined Hybrid score threshold for flagging pair plagiarism.",
        )

        lexical_threshold = st.slider(
            "Lexical Sensitivity Threshold",
            0.0,
            1.0,
            value=0.50,
            step=0.01,
            help="Direct word-for-word and N-gram match threshold.",
        )

        semantic_threshold = st.slider(
            "Semantic Sensitivity Threshold",
            0.0,
            1.0,
            value=0.65,
            step=0.01,
            help="Transformer embedding vector similarity threshold.",
        )

        if st.button("💾 Save Thresholds", type="primary"):
            st.session_state["threshold_slider"] = threshold
            st.success("✅ Thresholds saved to session state.")

    with tab_ocr:
        st.subheader("Optical Character Recognition (OCR)")
        st.caption("Used only for scanned or image-only PDF pages.")

        from src.core.document_parser import DEFAULT_OCR_DPI, SUPPORTED_OCR_LANGUAGES

        ocr_language_labels = {
            display_name: code for code, display_name in SUPPORTED_OCR_LANGUAGES.items()
        }

        selected_lang = st.selectbox(
            "OCR Language", options=list(ocr_language_labels.keys()), index=0
        )

        ocr_dpi = st.slider(
            "OCR DPI Resolution",
            min_value=150,
            max_value=400,
            value=DEFAULT_OCR_DPI,
            step=25,
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
                    )
                else:
                    st.download_button(
                        "⬇️ Download Raw Database",
                        data=snapshot,
                        file_name="corpus.db",
                        mime="application/vnd.sqlite3",
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
        )


if __name__ == "__main__":
    render_settings()
