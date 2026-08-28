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

"""Centralized Session State Keys for the Streamlit Application."""

from enum import Enum


class SessionKeys(str, Enum):
    """Every key this application stores in ``st.session_state``.

    The enum subclasses :class:`str` so a member can be used anywhere a plain
    string key is expected -- ``st.session_state[SessionKeys.LANG]`` and
    ``st.session_state["lang"]`` address the same slot, and a member can be
    passed straight to a widget's ``key=`` argument.

    Member values are the lower-cased member names, which keeps the mapping
    obvious and makes the migration from the old bare-string keys mechanical.
    """

    SESSION_ID = "session_id"
    AUTHENTICATED = "authenticated"
    USERNAME = "username"
    ROLE = "role"
    PDF_PASSWORDS = "pdf_passwords"
    LANG = "lang"
    MODEL_LOAD_TIME = "model_load_time"
    LAST_INTERACTION = "last_interaction"
    PENDING_2FA = "pending_2fa"
    PENDING_USERNAME = "pending_username"
    PENDING_ROLE = "pending_role"
    ANALYSIS_RESULTS = "analysis_results"
    ANALYSIS_FILE_SIGNATURE = "analysis_file_signature"
    DRIVE_FILES_DICT = "drive_files_dict"
    FAILED_DOCUMENTS = "failed_documents"
    WARNING_PAGE = "warning_page"
    SCANNING = "scanning"
    SHOW_TOUR = "show_tour"
    SELECTED_DOCUMENT_ID = "selected_document_id"
    SENT_ALERTS = "sent_alerts"
    AUDIT_REPORT_GENERATED = "audit_report_generated"
    INCIDENT_STREAM_AUTO_REFRESH = "incident_stream_auto_refresh"
    WARNINGS_EXPAND_ALL = "warnings_expand_all"
    LANG_SELECTOR = "lang_selector"
    THRESHOLD_SLIDER = "threshold_slider"
    LEXICAL_THRESHOLD_SLIDER = "lexical_threshold_slider"
    SEMANTIC_THRESHOLD_SLIDER = "semantic_threshold_slider"
    CHUNK_MATRIX_CHECKBOX = "chunk_matrix_checkbox"
    FAISS_TOP_K_SLIDER = "faiss_top_k_slider"
    CHUNK_SIZE_SLIDER = "chunk_size_slider"
    CHUNK_OVERLAP_SLIDER = "chunk_overlap_slider"
    OCR_LANGUAGE_SELECTOR = "ocr_language_selector"
    OCR_DPI_SLIDER = "ocr_dpi_slider"
    CLASS_FILTER_SELECTBOX = "class_filter_selectbox"
    AUDIT_LOG_PAGE = "audit_log_page"
    FORCE_DARK_CHARTS = "force_dark_charts"
    SESSION_START_TIME = "session_start_time"
    COMPACT_VIEW = "compact_view"
    ACCENT_COLOR = "accent_color"

    def __str__(self) -> str:  # pragma: no cover - convenience for f-strings/logging
        return self.value
