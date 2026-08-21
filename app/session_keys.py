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

    def __str__(self) -> str:  # pragma: no cover - convenience for f-strings/logging
        return self.value
