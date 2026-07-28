import os
import sqlite3
import sys
from pathlib import Path

# Fix Streamlit import paths by pointing to project root
FILE_PATH = Path(__file__).resolve()
ROOT_DIR = FILE_PATH.parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import base64
import html

# Standard / Third-party imports
import time

import _io
import psutil
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

from src.core.ai_detector import detect_documents_ai_probability
from src.core.embedding_model import embed_documents
from src.core.text_chunking import chunk_documents

load_dotenv()

from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from app.components.faiss_results import faiss_results_dataframe
from src.security.metadata_stripper import strip_exif_metadata
from src.utils.filename import sanitize_filename, unique_filename

try:
    from streamlit_plotly_events import plotly_events
except ImportError:  # pragma: no cover - optional dependency
    plotly_events = None

import logging

logger = logging.getLogger(__name__)
# Validate required environment variables during application startup
REQUIRED_ENV_VARS = [
    "REDIS_URL",
    "PLAGIARISM_WEBHOOK_URL",
    "API_BEARER_TOKEN",
]

missing_env_vars = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]

if missing_env_vars:
    logger.warning(
        "Missing environment variables: %s. "
        "Some features may not work correctly. "
        "Please configure them in your .env file.",
        ", ".join(missing_env_vars),
    )


from app.css_constants import (
    CLASS_CLEAR_ALL_CONTAINER,
    CLASS_SKELETON,
    CLASS_SKELETON_CHART,
    CLASS_SKELETON_METRIC,
    CLASS_SKELETON_TABLE,
    CLASS_SKELETON_TEXT,
    CLASS_SKELETON_TEXT_SHORT,
    CLASS_SKELETON_TITLE,
)
from app.theme import (
    back_to_top_html,
    empty_state_html,
    get_colors,
    get_theme_name,
    inject_css,
    pipeline_progress_html,
    set_theme,
    version_check_widget_html,
)
from src.core.app_config import get_app_title
from src.core.config import DEFAULT_THRESHOLDS, PLAGIARISM_THRESHOLD, severity_key
from src.core.document_parser import (
    DEFAULT_OCR_DPI,
    DEFAULT_OCR_LANGUAGE,
    SUPPORTED_OCR_LANGUAGES,
    OCRDependencyError,
    extract_text,
    prepare_text_for_embedding,
    remove_ignore_phrases,
)
from src.core.faiss_index import (
    build_index,
    build_index_from_matrix,
    load_index,
    load_or_rebuild_index,
    save_index,
    search_similar_chunks,
)
from src.core.similarity import (
    document_similarity_matrix,
    find_most_similar_chunks,
    flag_plagiarism,
)
from src.core.webhook import dispatch_plagiarism_alert
from src.i18n.translator import _SUPPORTED_LANGUAGES, get_text
from src.visualization.network_graph import plot_similarity_network


class OCRFileBatchError(Exception):
    """Exception raised when OCR extraction fails on one or more files in a batch."""

    def __init__(self, failed_files: list[str], failure_details: list[str]):
        self.failed_files = failed_files
        self.failure_details = failure_details
        super().__init__(f"OCR failed for files: {failed_files}")


from src.core.export_engine import LMSExportEngine
from src.core.synchronization import verify_and_repair_index
from src.core.telemetry import TelemetryService
from src.db import (
    clear_all_data,
    delete_tag,
    empty_trash,
    get_all_documents,
    get_all_embeddings,
    get_all_tags,
    get_chunk_registry,
    get_deleted_documents,
    get_document_word_counts,
    get_unique_class_sections,
    init_corpus_db,
    permanently_delete_document,
    restore_document,
    soft_delete_document,
)
from src.db.auth import (
    authenticate_user,
    check_login_rate_limit,
    clear_login_attempts,
    disable_2fa,
    enable_2fa,
    get_2fa_status,
    get_all_users,
    get_tour_completed,
    get_user_preferences,
    get_user_role,
    init_db,
    is_user_active,
    record_failed_login,
    set_tour_completed,
    set_user_active_status,
    update_user_preferences,
    verify_user,
)
from src.db.database_backup import create_corpus_database_snapshot
from src.db.incidents import (  # noqa: E402
    get_all_incidents_above_threshold_for_export,
    get_high_severity_trends,
    get_most_plagiarized_documents,
    sync_flagged_incidents,
)
from src.utils.diff_highlighter import highlight_overlap
from src.utils.excel_export import export_similarity_matrix_to_excel
from src.utils.pdf_report import highlight_pdf_matches  # noqa: E402
from src.utils.processing_time import (
    estimate_processing_seconds,
    uploaded_files_total_bytes,
)
from src.utils.redis_cache import (
    cache_session_state,
    clear_session,
    get_analysis_results,
    get_faiss_index,
    get_session_state,
    get_upload_count,
    increment_upload_count,
    is_upload_rate_limited,
)
from src.utils.warning_list import render_warning_controls
from src.visualization.analytics import (
    plot_document_sizes,
    plot_high_severity_trends,
    plot_most_plagiarized_documents,
    plot_similarity_distribution,
)
from src.visualization.heatmap import plot_similarity_heatmap  # noqa: E402

# Safe import for PDF Highlighting

try:

    from src.utils.pdf_highlighter import highlight_pdf_matches  # type: ignore
except Exception:
    highlight_pdf_matches = None

    # Safe import for Google Drive integration

    from src.utils.excel_export import export_similarity_matrix_to_excel
    from src.utils.json_export import export_similarity_matrix_to_json
except ImportError:

    from utils.excel_export import export_similarity_matrix_to_excel  # type: ignore
    from utils.json_export import export_similarity_matrix_to_json


# Initialize corpus database

try:
    from streamlit_tour import Tour
except ImportError:
    Tour = None


# Initialize databases

init_corpus_db()
init_db()

# Start lightweight REST API server for /healthz endpoint in background
import threading

import uvicorn

from src.api.app import app as fastapi_app


def _start_api_server():
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8000, log_level="warning")


threading.Thread(target=_start_api_server, daemon=True).start()

# Generate unique session ID for this Streamlit session
if "session_id" not in st.session_state:
    import uuid

    st.session_state.session_id = str(uuid.uuid4())

SESSION_ID = st.session_state.session_id

_BRANDING_CONFIG_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "branding_config.json")
)
_BRANDING_LOGO_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "branding_logo.png")
)
_INDEX_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "corpus.index")
)

# Startup Synchronization Check (Issue #361)
verify_and_repair_index(_INDEX_PATH)

from streamlit_tour import Tour

try:

    from streamlit_tour import Tour
except Exception:
    Tour = None

    from src.utils.google_drive import import_from_google_drive  # type: ignore
except Exception:
    import_from_google_drive = None

# -----------------------------------------------------------------------------
# Page Configuration & Session State
# -----------------------------------------------------------------------------


# Page Configuration
# NOTE: initial_sidebar_state="auto" lets Streamlit decide the sidebar's
# starting state based on viewport width. On screens narrower than the
# "md" breakpoint (768px) — phones and small tablets — the sidebar starts
# collapsed so it doesn't cover the similarity matrix / heatmap. On wider
# screens it behaves the same as "expanded". See issue #258.

APP_TITLE = get_app_title()

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="auto",
)


if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = None
if "pdf_passwords" not in st.session_state:
    st.session_state.pdf_passwords = {}
if "lang" not in st.session_state:
    st.session_state.lang = "en"

# -----------------------------------------------------------------------------
# Sidebar Settings Configuration
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ " + get_text("settings", lang=st.session_state.lang))

    selected_lang_name = st.selectbox(
        "🌐 Language / Idioma",
        options=["English", "Español"],
        index=0 if st.session_state.lang == "en" else 1,
    )
st.markdown(back_to_top_html(), unsafe_allow_html=True)
inject_css()

st.markdown(
    """
<style>
    .block-container { padding-top: 2rem; }
    .stAlert { border-radius: 8px; }
</style>
""",
    unsafe_allow_html=True,
)


# ── SESSION TIMEOUT & ROUTE PROTECTION ────────────────────────────────────────
TIMEOUT_LIMIT = 15 * 60  # 15 minutes in seconds

import streamlit.components.v1 as components

if st.session_state.get("authenticated", False):
    components.html(
        f"""
        <script>
            let timeoutLimit = {TIMEOUT_LIMIT} * 1000;
            let warningTime = timeoutLimit - (2 * 60 * 1000); // 2 minutes before
            let timer;
            let warningShown = false;

            function resetTimer() {{
                clearTimeout(timer);
                const warning = window.parent.document.getElementById('session-warning-toast');
                if (warning) {{
                    warning.style.display = 'none';
                }}
                warningShown = false;
                timer = setTimeout(showWarning, warningTime);
            }}

            function showWarning() {{
                if (warningShown) return;
                warningShown = true;
                let doc = window.parent.document;
                let warning = doc.getElementById('session-warning-toast');
                if (!warning) {{
                    warning = doc.createElement('div');
                    warning.id = 'session-warning-toast';
                    warning.style.position = 'fixed';
                    warning.style.top = '60px'; // below header
                    warning.style.right = '20px';
                    warning.style.backgroundColor = '#ffcc00';
                    warning.style.color = 'black';
                    warning.style.padding = '15px';
                    warning.style.borderRadius = '5px';
                    warning.style.zIndex = '9999';
                    warning.style.boxShadow = '0 4px 6px rgba(0,0,0,0.1)';
                    warning.innerHTML = '<strong>⚠️ Session Timeout Warning</strong><br>Your session will expire in 2 minutes due to inactivity. Please save your work or interact with the app to stay logged in.';
                    doc.body.appendChild(warning);
                }}
                warning.style.display = 'block';
            }}

            let parentDoc = window.parent.document;
            parentDoc.addEventListener('mousemove', resetTimer);
            parentDoc.addEventListener('keydown', resetTimer);
            parentDoc.addEventListener('scroll', resetTimer);
            parentDoc.addEventListener('click', resetTimer);

            resetTimer();
        </script>
        """,
        height=0,
    )


# 1. Handle Automatic Session Expiration (Inactivity Check)
cached_last_interaction = get_session_state(SESSION_ID, "last_interaction")
if cached_last_interaction is not None:
    last_interaction = cached_last_interaction
elif "last_interaction" in st.session_state:
    last_interaction = st.session_state.last_interaction
else:
    last_interaction = None

if last_interaction and st.session_state.get("authenticated", False):
    elapsed_time = time.time() - last_interaction
    if elapsed_time > TIMEOUT_LIMIT:
        for key in ["authenticated", "username", "role", "last_interaction"]:
            if key in st.session_state:
                del st.session_state[key]
        clear_session(SESSION_ID)
        from src.errors import UI_SESSION_EXPIRED

        st.warning(UI_SESSION_EXPIRED)
        st.stop()
    else:
        st.session_state.last_interaction = time.time()
        cache_session_state(SESSION_ID, "last_interaction", time.time())

# ── Handle OAuth Callback (GitHub / Google SSO) ──────────────────────────────
if not st.session_state.get("authenticated", False):
    if "code" in st.query_params and "state" in st.query_params:
        _code = st.query_params["code"]
        _state = st.query_params["state"]
        from src.db.auth import get_or_create_sso_user
        from src.utils.sso import exchange_github_code, exchange_google_code

        _user_info = None
        if _state.startswith("google_"):
            _user_info = exchange_google_code(_code)
        elif _state.startswith("github_"):
            _user_info = exchange_github_code(_code)
        if _user_info and _user_info.get("email"):
            _email = _user_info["email"]
            if not is_user_active(_email):
                st.error("🚨 Account suspended. Please contact your administrator.")
                st.query_params.clear()
            else:
                _role = get_or_create_sso_user(_email)
                st.session_state.authenticated = True
                st.session_state.username = _email
                st.session_state.role = _role
                st.session_state.last_interaction = time.time()
                cache_session_state(SESSION_ID, "authenticated", True)
                cache_session_state(SESSION_ID, "username", _email)
                cache_session_state(SESSION_ID, "role", _role)
                cache_session_state(SESSION_ID, "last_interaction", time.time())
                st.query_params.clear()
                st.rerun()
        else:
            st.error("🚨 SSO authentication failed. Could not retrieve your email.")
            st.query_params.clear()

# Render Login UI if not authenticated
if not st.session_state.get("authenticated", False):
    if st.session_state.get("pending_2fa", False):
        with st.form("otp_form"):
            st.subheader("🔒 Two-Factor Authentication")
            st.info(
                "Enter the 6-digit verification token from your Google Authenticator/Authy app."
            )
            otp_code = st.text_input(
                "Verification Code", max_chars=6, key="login_otp_code"
            )
            col1, col2 = st.columns(2)
            with col1:
                verify_submitted = st.form_submit_button(
                    "Verify", use_container_width=True
                )
            with col2:
                cancel_submitted = st.form_submit_button(
                    "Cancel", use_container_width=True
                )

            if verify_submitted:
                username = st.session_state.get("pending_username")
                enabled, otp_secret = get_2fa_status(username)
                if enabled and otp_secret:
                    import pyotp

                    totp = pyotp.TOTP(otp_secret)
                    if totp.verify(otp_code.strip()):
                        role = st.session_state.get("pending_role")
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        st.session_state.role = role
                        st.session_state.last_interaction = time.time()

                        cache_session_state(SESSION_ID, "authenticated", True)
                        cache_session_state(SESSION_ID, "username", username)
                        cache_session_state(SESSION_ID, "role", role)
                        cache_session_state(SESSION_ID, "last_interaction", time.time())
                        prefs = get_user_preferences(username)
                        st.session_state.threshold = prefs.get(
                            "threshold", DEFAULT_THRESHOLDS.plagiarism
                        )
                        st.session_state.telemetry_opt_in = prefs.get(
                            "telemetry_opt_in", True
                        )
                        from src.db.auth import get_user_theme

                        st.session_state.theme = get_user_theme(username)
                        set_theme(st.session_state.theme)

                        # Clear pending state
                        del st.session_state["pending_2fa"]
                        del st.session_state["pending_username"]
                        del st.session_state["pending_role"]

                        st.success(f"✅ Welcome back, {username}!")
                        st.rerun()
                    else:
                        st.error("🚨 Invalid verification code. Please try again.")
                else:
                    st.error("🚨 2FA configuration error. Please contact admin.")

            if cancel_submitted:
                del st.session_state["pending_2fa"]
                del st.session_state["pending_username"]
                del st.session_state["pending_role"]
                st.rerun()
        st.stop()

    with st.form("login_form"):
        username = st.text_input("Username", value="admin")
        password = st.text_input("Password", type="password", value="admin")
        login_submitted = st.form_submit_button("Log In", use_container_width=True)

        if login_submitted:
            username = username.strip().lower()
            prefs = get_user_preferences(username)
            st.session_state.threshold = prefs.get(
                "threshold", DEFAULT_THRESHOLDS.plagiarism
            )
            st.session_state.telemetry_opt_in = prefs.get("telemetry_opt_in", True)
            from src.db.auth import get_user_theme

            st.session_state.theme = get_user_theme(username)
            set_theme(st.session_state.theme)

            if not username or not password:
                from src.errors import AUTH_BLANK_CREDENTIALS

                st.error(f"🚨 {AUTH_BLANK_CREDENTIALS}")
            else:
                is_allowed, error_msg = check_login_rate_limit(username)
                if not is_allowed:
                    st.error(f"🚨 {error_msg}")
                elif not is_user_active(username):
                    st.error("🚨 Account suspended. Please contact your administrator.")
                elif verify_user(username, password):
                    role = get_user_role(username)
                    if role is None:
                        from src.errors import AUTH_ROLE_UNDETERMINED

                        st.error(f"🚨 {AUTH_ROLE_UNDETERMINED}")
                    else:
                        clear_login_attempts(username)
                        enabled, _ = get_2fa_status(username)
                        if enabled:
                            st.session_state.pending_2fa = True
                            st.session_state.pending_username = username
                            st.session_state.pending_role = role
                            st.rerun()
                        else:
                            st.session_state.authenticated = True
                            st.session_state.username = username
                            st.session_state.role = role
                            st.session_state.last_interaction = time.time()
                            cache_session_state(SESSION_ID, "authenticated", True)
                            cache_session_state(SESSION_ID, "username", username)
                            cache_session_state(SESSION_ID, "role", role)
                            cache_session_state(
                                SESSION_ID, "last_interaction", time.time()
                            )
                            st.success(f"Welcome back, {role.capitalize()}!")
                            st.success(f"✅ Welcome back, {role.capitalize()}!")
                            st.rerun()
                else:
                    # Record failed login attempt
                    record_failed_login(username)
                    from src.errors import AUTH_INVALID_CREDENTIALS

                    st.error(f"🚨 {AUTH_INVALID_CREDENTIALS}")

    # ── SSO Sign-In Options ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<p style='text-align:center;color:#888;font-size:0.85rem;'>or sign in with</p>",
        unsafe_allow_html=True,
    )
    _sso_col1, _sso_col2 = st.columns(2)
    with _sso_col1:
        if st.button(
            "🐙 Sign in with GitHub", use_container_width=True, key="github_sso_btn"
        ):
            from src.utils.sso import get_github_auth_url

            _github_url, _github_state = get_github_auth_url()
            st.session_state["sso_state"] = _github_state
            st.markdown(
                f"<meta http-equiv='refresh' content='0; url={_github_url}'>",
                unsafe_allow_html=True,
            )
    with _sso_col2:
        if st.button(
            "🔵 Sign in with Google", use_container_width=True, key="google_sso_btn"
        ):
            from src.utils.sso import get_google_auth_url

            _google_url, _google_state = get_google_auth_url()
            st.session_state["sso_state"] = _google_state
            st.markdown(
                f"<meta http-equiv='refresh' content='0; url={_google_url}'>",
                unsafe_allow_html=True,
            )
    st.stop()

# Active user role
user_role = st.session_state.get("role", "user")

# Sync threshold from URL query parameters (bi-directional)
if "threshold" in st.query_params:
    q_val_raw = st.query_params["threshold"]
    if st.session_state.get("last_seen_threshold_query") != q_val_raw:
        try:
            q_threshold = float(q_val_raw)
            if 0.0 <= q_threshold <= 1.0:
                st.session_state.threshold_slider = q_threshold
                st.session_state.threshold = q_threshold
                st.session_state.last_seen_threshold_query = q_val_raw
        except ValueError:
            pass
elif "threshold_slider" not in st.session_state:
    st.session_state.threshold_slider = st.session_state.get(
        "threshold", DEFAULT_THRESHOLDS.plagiarism
    )


# Resolve fallback configuration variables (ensuring all roles have access to these settings)
selected_lang_name = st.session_state.get("lang_selector", "English")
lang_code = "es" if selected_lang_name == "Español" else "en"

threshold = st.session_state.get("threshold_slider", DEFAULT_THRESHOLDS.plagiarism)
faiss_top_k = st.session_state.get("faiss_top_k_slider", 5)
use_chunk_matrix = st.session_state.get("chunk_matrix_checkbox", False)
chunk_size = st.session_state.get("chunk_size_slider", 500)
chunk_overlap = st.session_state.get("chunk_overlap_slider", 50)
ignore_phrases = st.session_state.get("ignore_phrases_textarea", "")
ocr_language_selector_val = st.session_state.get(
    "ocr_language_selector", DEFAULT_OCR_LANGUAGE
)
ocr_language_map = {
    "English": "eng",
    "Español": "spa",
    "Français": "fra",
    "eng": "eng",
    "spa": "spa",
    "fra": "fra",
}
ocr_language = ocr_language_map.get(ocr_language_selector_val, DEFAULT_OCR_LANGUAGE)
ocr_dpi = st.session_state.get("ocr_dpi_slider", DEFAULT_OCR_DPI)
heatmap_cmap = "OrRd"

unique_classes = ["All Classes"] + get_unique_class_sections()
selected_class = st.session_state.get("class_filter_selectbox", "All Classes")


@st.dialog("⚠️ Confirm Bulk Clear")
def clear_all_dialog():
    st.markdown(
        "**WARNING:** This action is destructive and cannot be undone. "
        "This will permanently delete all student documents, paragraph chunks, "
        "and plagiarism incidents from the database, and reset the FAISS index."
    )
    st.write("Are you absolutely sure you want to proceed?")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Cancel", use_container_width=True, key="cancel_clear_all"):
            st.rerun()
    with col2:
        if st.button(
            "Clear All",
            type="primary",
            use_container_width=True,
            key="confirm_clear_all",
        ):
            clear_all_data()
            if os.path.exists(_INDEX_PATH):
                try:
                    os.remove(_INDEX_PATH)
                except OSError as e:
                    print(f"Error removing FAISS index: {e}")
                except Exception as e:
                    logger.error(f"Error removing FAISS index: {e}")

            try:
                from src.utils.redis_cache import get_cache

                cache = get_cache()
                if cache.is_available():
                    cache.delete("faiss:index:corpus_index")
                    cache.clear_pattern("analysis:*")
            except (ImportError, RuntimeError, ConnectionError) as e:
                print(f"Error invalidating cache: {e}")
            except Exception as e:
                logger.error(f"Error invalidating cache: {e}")

            if "analysis_results" in st.session_state:
                st.session_state.analysis_results = None
            if "analysis_file_signature" in st.session_state:
                st.session_state.analysis_file_signature = None
            if "processed_pipeline_signature" in st.session_state:
                st.session_state.processed_pipeline_signature = None

            st.success("✅ All documents, chunks, and incidents have been cleared.")
            st.rerun()


# ── Top-right Theme Toggle ───────────────────────────────────────────────────
current_theme = get_theme_name()
_, theme_col = st.columns([0.94, 0.06])

with theme_col:
    theme_icon = "☀️" if current_theme == "Dark" else "🌙"
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        _ctx = get_script_run_ctx()
        if _ctx and _ctx.current_form_id:
            _ctx.current_form_id = ""
    except Exception:
        pass
    if st.button(theme_icon, key="theme_toggle"):
        new_theme = "Light" if current_theme == "Dark" else "Dark"
        set_theme(new_theme)
        st.rerun()


# ── Sidebar (ROLE RESTRICTED Settings & i18n) ─────────────────────────────────


def save_preferences_callback():
    if "username" in st.session_state:
        prefs = {
            "threshold": st.session_state.get(
                "threshold_slider", DEFAULT_THRESHOLDS.plagiarism
            ),
            "telemetry_opt_in": st.session_state.get("telemetry_opt_in_toggle", True),
        }
        update_user_preferences(st.session_state.username, prefs)

        from src.db.auth import set_user_theme

        theme_val = st.session_state.get("theme_selector", "Light")
        set_user_theme(st.session_state.username, theme_val)


with st.sidebar:
    st.markdown(f"👤 Logged in as **{st.session_state.get('username', '')}**")

    # Render cached telemetry user count badge (only if the user has opted in)
    if st.session_state.get("telemetry_opt_in", True):
        try:
            active_users = TelemetryService.get_active_user_count()
            st.caption(f"Total System Users: {active_users}")
        except Exception:
            pass

    if st.button("🚪 Log Out", use_container_width=True):
        import logging
        from datetime import datetime, timezone

        logger = logging.getLogger(__name__)
        username = st.session_state.get("username", "unknown")
        timestamp = datetime.now(timezone.utc).isoformat()
        logger.info("User '%s' logged out at %s", username, timestamp)

        for key in ["authenticated", "username", "role"]:
            if key in st.session_state:
                del st.session_state[key]
        clear_session(SESSION_ID)
        st.rerun()

    if user_role == "admin":
        st.markdown("---")
        st.markdown("### 📁 Document Management")
        existing_docs = get_all_documents()
        session_uploaded_docs = st.session_state.get("session_uploaded_docs", set())

        doc_filter = st.text_input(
            "Filter documents by filename", key="doc_mgmt_filter"
        )

        if existing_docs:
            filtered_docs = [
                d
                for d in existing_docs
                if not doc_filter or doc_filter.lower() in str(d["filename"]).lower()
            ]
            st.write(f"**{len(filtered_docs)}** documents matching")

            items_per_page = 20
            total_pages = max(1, (len(filtered_docs) - 1) // items_per_page + 1)

            current_page = st.session_state.get("sidebar_doc_page", 1)
            if current_page > total_pages:
                current_page = total_pages
                st.session_state.sidebar_doc_page = current_page

            start_idx = (current_page - 1) * items_per_page
            end_idx = start_idx + items_per_page

            for doc in filtered_docs[start_idx:end_idx]:
                st.markdown('<div class="doc-row">', unsafe_allow_html=True)
                col1, col2 = st.columns([3, 1])
                with col1:
                    is_new = doc["filename"] in session_uploaded_docs
                    badge_html = (
                        ' <span style="background-color:#28a745;color:white;font-size:0.7rem;padding:2px 6px;border-radius:4px;font-weight:bold;">NEW</span>'
                        if is_new
                        else ""
                    )
                    safe_display_name = html.escape(
                        str(doc["filename"]),
                        quote=True,
                    )
                    st.markdown(
                        f"📄 {safe_display_name}{badge_html}",
                        unsafe_allow_html=True,
                    )
                with col2:
                    if st.button("🗑️", key=f"del_{doc['filename']}"):
                        st.session_state._pending_delete = doc["filename"]
                        st.rerun()

            if total_pages > 1:
                p_col1, p_col2, p_col3 = st.columns([1, 2, 1])
                with p_col1:
                    if st.button(
                        "Prev", disabled=(current_page == 1), key="prev_doc_page"
                    ):
                        st.session_state.sidebar_doc_page = current_page - 1
                        st.rerun()
                with p_col2:
                    st.markdown(
                        f"<div style='text-align: center; margin-top: 5px;'><small>Page {current_page}/{total_pages}</small></div>",
                        unsafe_allow_html=True,
                    )
                with p_col3:
                    if st.button(
                        "Next",
                        disabled=(current_page == total_pages),
                        key="next_doc_page",
                    ):
                        st.session_state.sidebar_doc_page = current_page + 1
                        st.rerun()

            pending = st.session_state.get("_pending_delete")
            if pending:
                st.markdown("---")
                st.warning(f"Are you sure you want to delete **{pending}**?")
                confirm_col, cancel_col = st.columns(2)
                with confirm_col:
                    if st.button(
                        "Yes, delete", type="primary", key="confirm_delete_doc"
                    ):
                        soft_delete_document(pending)
                        try:
                            from src.utils.redis_cache import get_cache

                            cache = get_cache()
                            if cache.is_available():
                                cache.delete("faiss:index:corpus_index")
                                cache.clear_pattern("analysis:*")
                        except Exception as e:
                            logger.error(f"Error invalidating cache: {e}")

                        if "analysis_results" in st.session_state:
                            st.session_state.analysis_results = None
                        if "processed_pipeline_signature" in st.session_state:
                            st.session_state.processed_pipeline_signature = None

                        embeddings_matrix = get_all_embeddings()
                        if embeddings_matrix.size > 0:
                            new_index = build_index_from_matrix(embeddings_matrix)
                            save_index(new_index, _INDEX_PATH)
                        else:
                            if os.path.exists(_INDEX_PATH):
                                os.remove(_INDEX_PATH)
                        del st.session_state._pending_delete
                        st.rerun()
                with cancel_col:
                    if st.button("Cancel", key="cancel_delete_doc"):
                        del st.session_state._pending_delete
                        st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

        # ── Manage Tags ───────────────────────────────────────────────────────
        with st.expander("🏷️ Manage Tags", expanded=False):
            all_tags = get_all_tags()
            if not all_tags:
                st.caption("No tags found in the database.")
            else:
                for tag in all_tags:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"`{tag}`")
                    with col2:
                        if st.button("Delete", key=f"del_tag_{tag}", type="secondary"):
                            affected = delete_tag(tag)
                            st.success(
                                f"Deleted tag `{tag}` from {affected} document(s)."
                            )
                            st.rerun()

        # ── Generate Mock Data (Issue #255) ───────────────────────────────────
        # Hidden developer utility: generates 5 fake essays via Faker so the
        # app is immediately usable after cloning without manual PDF uploads.
        with st.expander("🧪 Developer Tools", expanded=False):
            st.caption(
                "Generate fake student essays to populate the corpus and preview "
                "the app without uploading real PDFs."
            )
            mock_class = st.text_input(
                "Mock Class/Section",
                value="Demo Class",
                key="mock_class_input",
                help="Class section label assigned to all generated essays.",
            )
            mock_assignment = st.text_input(
                "Mock Assignment Title",
                value="Demo Assignment",
                key="mock_assignment_input",
                help="Assignment title assigned to all generated essays.",
            )
            if st.button(
                "⚗️ Generate Mock Data",
                key="generate_mock_data_button",
                use_container_width=True,
                help="Creates 5 fake student essays using the Faker library, "
                "stores them in corpus.db, and rebuilds the FAISS index.",
            ):
                try:
                    from src.utils.mock_data import generate_mock_data as _gen_mock

                    with st.spinner(
                        "⚗️ Generating mock essays and building FAISS index…"
                    ):
                        result = _gen_mock(
                            num_essays=5,
                            class_section=mock_class.strip() or "Demo Class",
                            assignment_title=mock_assignment.strip()
                            or "Demo Assignment",
                            chunk_size=st.session_state.get("chunk_size_slider", 500),
                            chunk_overlap=st.session_state.get(
                                "chunk_overlap_slider", 50
                            ),
                        )

                    added = result["essays"]
                    skipped = result["skipped"]
                    ntotal = result["faiss_ntotal"]

                    if added:
                        st.success(
                            f"✅ Added **{len(added)}** mock essay(s): "
                            + ", ".join(name for _, name in added)
                        )
                    if skipped:
                        st.info(
                            f"ℹ️ {len(skipped)} essay(s) already existed and were skipped."
                        )
                    st.success(
                        f"🗂️ FAISS index rebuilt with **{ntotal}** total vectors."
                    )
                    # Invalidate cached analysis so the UI reloads with new docs
                    st.session_state.analysis_results = None
                    st.rerun()

                except ImportError:
                    st.error(
                        "❌ The `faker` package is not installed. "
                        "Run `pip install faker` and restart the app."
                    )
                except (ValueError, RuntimeError, TypeError, OSError) as _mock_err:
                    st.error(f"❌ Mock data generation failed: {_mock_err}")

        st.markdown(
            f'<div class="{CLASS_CLEAR_ALL_CONTAINER}">', unsafe_allow_html=True
        )
        if st.button(
            "🗑️ Clear All Documents",
            key="clear_all_documents_button",
            use_container_width=True,
        ):
            clear_all_dialog()
        st.markdown("</div>", unsafe_allow_html=True)

    else:
        threshold = PLAGIARISM_THRESHOLD
        use_chunk_matrix = False
        faiss_top_k = 5
        chunk_size = 500
        chunk_overlap = 50
        ocr_language = DEFAULT_OCR_LANGUAGE
        ocr_dpi = DEFAULT_OCR_DPI

    st.markdown("---")
    unique_classes = ["All Classes"] + get_unique_class_sections()

    selected_class = st.selectbox("Select Class/Section", unique_classes, index=0)

# ── Main UI ───────────────────────────────────────────────────────────────────
st.title(f"🔍 {APP_TITLE}")

uploaded_files = st.file_uploader(
    "📂 Upload Assignments (PDF, DOCX, DOC, TXT, ZIP, PNG, JPG)",
    type=["pdf", "docx", "doc", "txt", "zip", "png", "jpg", "jpeg"],
    accept_multiple_files=True,
    key="file_uploader",
)
# ── MAIN APPLICATION SECTIONS (ROLE CHECKED) ──────────────────────────────────

if user_role != "admin":
    # STANDARD USER VIEW: Student Query / Search Panel Only (No admin PDF uploading)
    st.subheader("🔎 Secure Student Search Portal")
    st.caption(
        "Paste a text snippet below to check its similarity against existing indexed assignments."
    )

    st.info(
        "🔒 Note: Direct assignment uploads and detailed breakdown panels are restricted to Administrator access. Your queries are anonymized for privacy."
    )

    query_text = st.text_area(
        "Paste a text snippet to check against index:",
        height=150,
        placeholder="Paste a paragraph here to check for plagiarism...",
    )

    if st.button("🔍 Run Quick Verification", key="user_query") and query_text.strip():
        # Load existing index and registry from database
        from src.core.faiss_index import build_index_from_matrix
        from src.db.corpus_db import get_all_embeddings, get_chunk_registry

        with st.spinner("Loading index and searching..."):
            try:
                registry = get_chunk_registry()
                embeddings_matrix = get_all_embeddings()

                if embeddings_matrix.shape[0] == 0:
                    from src.errors import UI_NO_DOCUMENTS_INDEXED

                    st.warning(UI_NO_DOCUMENTS_INDEXED)
                else:
                    # Build index from stored embeddings
                    faiss_index = build_index_from_matrix(
                        embeddings_matrix, index_type="auto"
                    )

                    # Embed the query
                    from src.core.embedding_model import embed_chunks

                    query_vec = embed_chunks([query_text.strip()])[0]

                    # Search with threshold
                    faiss_threshold = threshold
                    results = search_similar_chunks(
                        query_vec,
                        faiss_index,
                        registry,
                        top_k=faiss_top_k,
                        threshold=faiss_threshold,
                    )

                    if not results:
                        st.success(
                            "✅ No significant matches found in the assignment database."
                        )
                    else:
                        st.success(
                            f"Found **{len(results)}** potentially similar passages."
                        )

                        # Anonymize document names
                        doc_id_map = {}
                        anon_counter = 1

                        for record, score in results:
                            if record.doc_name not in doc_id_map:
                                doc_id_map[record.doc_name] = (
                                    f"Document-{anon_counter:03d}"
                                )
                                anon_counter += 1

                        # Display anonymized results
                        for rank, (record, score) in enumerate(results, 1):
                            anon_doc_name = doc_id_map[record.doc_name]
                            color = "#ff4b4b" if score >= 0.90 else "#ffa500"

                            with st.expander(
                                f"#{rank} · {anon_doc_name} (chunk #{record.chunk_index+1}) "
                                f"— {score:.1%}",
                                expanded=(rank == 1),
                            ):
                                cq, cm = st.columns(2)
                                with cq:
                                    st.markdown("**Your query:**")
                                    st.info(query_text.strip())
                                with cm:
                                    st.markdown(
                                        f"**Matching passage in {anon_doc_name}:**"
                                    )
                                    st.warning(record.chunk_text)

                                st.markdown(
                                    f"<div style='text-align:right;'>"
                                    f"<span style='background:{color};color:white;padding:3px 12px;"
                                    f"border-radius:10px;font-size:0.85rem;font-weight:700;'>"
                                    f"Similarity: {score*100:.1f}%</span></div>",
                                    unsafe_allow_html=True,
                                )

                        st.caption(
                            "🔒 Document names are anonymized to protect student privacy."
                        )

            except Exception as e:
                from src.errors import ui_index_load_failed

                st.error(ui_index_load_failed(error=str(e)))
                st.info(
                    "Please ensure documents have been indexed by an administrator."
                )
else:
    # ADMINISTRATOR ACCESS: Full Upload Pipeline & Evaluation Dashboards

    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = None

    # Load or initialize FAISS index
    if os.path.exists(_INDEX_PATH):
        faiss_index = load_index(_INDEX_PATH)
        registry = get_chunk_registry()
        if faiss_index is not None and faiss_index.ntotal != len(registry):
            all_embs = get_all_embeddings()
            if len(all_embs) > 0 and len(all_embs) == len(registry):
                faiss_index = build_index_from_matrix(all_embs)
                save_index(faiss_index, _INDEX_PATH)
            elif len(all_embs) == 0:
                faiss_index = None
                registry = []
        if faiss_index is not None:
            st.info(f"📂 Loaded existing FAISS index with {faiss_index.ntotal} vectors")
    else:
        threshold = DEFAULT_THRESHOLDS.plagiarism
        use_chunk_matrix = False
        faiss_top_k = 5
        chunk_size = 500
        chunk_overlap = 50
        ocr_language = DEFAULT_OCR_LANGUAGE
        ocr_dpi = DEFAULT_OCR_DPI
        ignore_phrases = ""
        st.info("ℹ️ Settings configuration is restricted to Administrators.")

# ── Onboarding Tour for First-Time Admin Users ───────────────────────────────────
if (
    Tour is not None
    and user_role == "admin"
    and not get_tour_completed(st.session_state.username)
):
    username = st.session_state.username

    if st.button("🎯 Start Guided Tour", key="start_tour_button", type="primary"):
        st.session_state.show_tour = True

    if st.session_state.get("show_tour", False):
        tour_steps = [
            Tour.info(
                title="👋 Welcome to the Plagiarism Detection System!",
                desc="This guided tour will walk you through the key features to help you get started.",
            ),
            Tour.bind(
                "threshold_slider",
                title="⚙️ Plagiarism Threshold",
                desc=f"Adjust the flagging threshold. Medium severity starts at {DEFAULT_THRESHOLDS.medium:.0%} and High at {DEFAULT_THRESHOLDS.high:.0%}.",
                side="right",
            ),
            Tour.bind(
                "class_filter_selectbox",
                title="🔍 Class Filter",
                desc="Filter analysis results by specific class sections.",
                side="right",
            ),
            Tour.info(
                title="📊 Analysis Dashboard",
                desc="View similarity metrics, flagged pairs, and comparisons in the tabs below.",
            ),
            Tour.info(
                title="🎉 You're All Set!",
                desc="You can now start uploading assignments and detecting plagiarism.",
            ),
        ]

        tour = Tour(steps=tour_steps)
        tour.start()

        if st.button("✅ Finish Tour", use_container_width=True):
            set_tour_completed(username, True)
            st.session_state.show_tour = False
            st.success("✅ Onboarding tour completed!")
            st.rerun()

# ── Main Header ──────────────────────────────────────────────────────────────
configured_app_title = os.getenv("APP_TITLE", "").strip()
if configured_app_title:
    st.title(f"🔍 {APP_TITLE}")
else:
    st.title(get_text("title", lang=lang_code))
st.markdown(get_text("subtitle", lang=lang_code))
st.divider()

# ── MAIN APPLICATION SECTIONS ──────────────────────────────────────────────────
if user_role != "admin":
    # STANDARD USER VIEW
    st.subheader("🔎 Secure Student Search Portal")
    st.caption(
        "Paste a text snippet below to check its similarity against existing indexed assignments."
    )
    st.info(
        "🔒 Note: Direct assignment uploads are restricted to Administrator access."
    )
    query_text = st.text_area(
        "Search Query Text:",
        placeholder="Paste document content here to search for matching plagiarism...",
        height=200,
    )

    if st.button("🔍 Run Quick Verification", key="user_query") and query_text.strip():

        with st.spinner("Loading index and searching..."):
            try:
                registry = get_chunk_registry()
                embeddings_matrix = get_all_embeddings()

                if embeddings_matrix.shape[0] == 0:
                    st.warning("No documents are currently indexed.")
                else:
                    memory = psutil.virtual_memory()
                    if memory.percent >= 85:
                        st.warning(
                            "⚠️ High memory usage detected (>85%). Large FAISS indexes may cause system instability or out-of-memory crashes."
                        )
                    faiss_index = build_index_from_matrix(
                        embeddings_matrix, index_type="auto"
                    )
                    processed_query = query_text.strip()
                    query_vec = embed_chunks([processed_query])[0]
                    faiss_threshold = 0.50  # Standard user default

                    results = search_similar_chunks(
                        query_vec,
                        faiss_index,
                        registry,
                        top_k=5,
                        threshold=faiss_threshold,
                    )

                    if not results:
                        st.success(
                            "✅ No significant matches found in the assignment database."
                        )
                    else:
                        st.success(
                            f"✅ Found **{len(results)}** potentially similar passages."
                        )

                        doc_id_map = {}
                        anon_counter = 1

                        for record, score in results:
                            if record.doc_name not in doc_id_map:
                                doc_id_map[record.doc_name] = (
                                    f"Document-{anon_counter:03d}"
                                )
                                anon_counter += 1

                        for rank, (record, score) in enumerate(results, 1):
                            anon_doc_name = doc_id_map[record.doc_name]
                            color = "#ff4b4b" if score >= 0.90 else "#ffa500"

                            with st.expander(
                                f"#{rank} · {anon_doc_name} (chunk #{record.chunk_index+1}) — {score:.1%}",
                                expanded=(rank == 1),
                            ):
                                cq, cm = st.columns(2)
                                with cq:
                                    st.markdown("**Your query:**")
                                    st.info(query_text.strip())
                                with cm:
                                    st.markdown(
                                        f"**Matching passage in {anon_doc_name}:**"
                                    )
                                    st.warning(record.chunk_text)

                                st.markdown(
                                    f"<div style='text-align:right;'>"
                                    f"<span style='background:{color};color:white;padding:3px 12px;"
                                    f"border-radius:10px;font-size:0.85rem;font-weight:700;'>"
                                    f"Similarity: {score*100:.1f}%</span></div>",
                                    unsafe_allow_html=True,
                                )

                        st.caption(
                            "🔒 Document names are anonymized to protect student privacy."
                        )
            except (RuntimeError, ValueError, OSError, TypeError) as e:
                st.error(f"🚨 Error loading index: {str(e)}")
else:
    # ADMIN FULL ACCESS VIEW
    faiss_index = None
    registry = []

    cached_index_data = get_faiss_index("corpus_index")
    if cached_index_data is not None:
        try:
            import faiss

            index_buffer = _io.BytesIO(cached_index_data)
            faiss_index = faiss.deserialize_index(faiss.read_index(index_buffer))
            registry = get_chunk_registry()
            st.info(
                f"📂 Loaded FAISS index from Redis cache with {faiss_index.ntotal} vectors"
            )
        except (RuntimeError, ValueError, OSError) as e:
            print(f"[Redis] Error loading cached index: {e}, falling back to disk")
        except Exception as e:
            logger.warning(
                f"[Redis] Error loading cached index: {e}, falling back to disk"
            )

    if faiss_index is None:
        try:
            memory = psutil.virtual_memory()
            if memory.percent >= 85:
                st.warning(
                    "⚠️ High memory usage detected (>85%). Large FAISS indexes may cause system instability or out-of-memory crashes."
                )
            faiss_index, registry, index_recovered = load_or_rebuild_index(_INDEX_PATH)
            if index_recovered:
                if faiss_index.ntotal:
                    st.warning(
                        f"FAISS index rebuilt from {faiss_index.ntotal} stored vectors."
                    )
                else:
                    st.info(
                        "No stored embeddings found. An empty FAISS index was initialized."
                    )
            else:
                st.info(
                    f"Loaded existing FAISS index with {faiss_index.ntotal} vectors."
                )
        except (RuntimeError, ValueError, OSError):
            faiss_index = None
            registry = []

    def load_analysis_results_from_db():
        import numpy as np
        import pandas as pd

        docs = get_all_documents()
        if not docs:
            return None

        raw_texts = {}
        chunked_docs = {}
        embeddings = {}

        try:
            from src.db.corpus_db import _connect

            with _connect() as conn:
                rows = conn.execute(
                    "SELECT filename, chunk_index, chunk_text, embedding FROM chunks ORDER BY filename, chunk_index"
                ).fetchall()

            for fname, chunk_idx, text, emb_blob in rows:
                if fname not in raw_texts:
                    raw_texts[fname] = ""
                    chunked_docs[fname] = []
                    embeddings[fname] = []

                raw_texts[fname] += text + " "
                chunked_docs[fname].append(text)

                emb = np.frombuffer(emb_blob, dtype=np.float32)
                embeddings[fname].append(emb)

            # Convert lists to numpy arrays
            for fname in embeddings:
                embeddings[fname] = np.vstack(embeddings[fname])

            sim_df = document_similarity_matrix(embeddings)

            names = list(embeddings.keys())
            n = len(names)
            chunk_mat = np.zeros((n, n))
            for i, na in enumerate(names):
                for j, nb in enumerate(names):
                    if i == j:
                        chunk_mat[i, j] = 1.0
                    elif j > i:
                        ea, eb = embeddings[na], embeddings[nb]
                        score = (
                            float(np.max(cosine_similarity(ea, eb)))
                            if ea.size and eb.size
                            else 0.0
                        )
                        chunk_mat[i, j] = score
                        chunk_mat[j, i] = score
            chunk_sim_df = pd.DataFrame(chunk_mat, index=names, columns=names)

            f_index = load_index(_INDEX_PATH) if os.path.exists(_INDEX_PATH) else None
            f_registry = get_chunk_registry()

            # Default AI probabilities for loaded documents to 0.0
            ai_probs = {
                fname: {"overall": 0.0, "max": 0.0, "chunk_scores": []}
                for fname in names
            }

            return (
                raw_texts,
                chunked_docs,
                embeddings,
                sim_df,
                chunk_sim_df,
                f_index,
                f_registry,
                ai_probs,
            )
        except (RuntimeError, ValueError, OSError, TypeError, KeyError) as err:
            print(f"Error rebuilding analysis results from DB: {err}")
        except Exception as err:
            logger.error(f"Error rebuilding analysis results from DB: {err}")
            return None

    if (
        "analysis_results" not in st.session_state
        or st.session_state.analysis_results is None
    ):
        st.session_state.analysis_results = None
        cached_results = get_analysis_results(f"{SESSION_ID}:current")
        if cached_results is not None:
            st.session_state.analysis_results = cached_results
        else:
            st.session_state.analysis_results = load_analysis_results_from_db()

    if "analysis_file_signature" not in st.session_state:
        st.session_state.analysis_file_signature = None

        cached_signature = get_session_state(SESSION_ID, "analysis_file_signature")
        if cached_signature is not None:
            st.session_state.analysis_file_signature = cached_signature

    if "failed_documents" not in st.session_state:
        st.session_state.failed_documents = {}

    # 1. LOCAL FILE UPLOADER (Dynamic Title Translation)
    uploaded_files = st.file_uploader(
        get_text("upload_title", lang=lang_code),
        type=["pdf", "docx", "doc", "txt", "zip", "csv", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="file_uploader",
    )

    if uploaded_files:
        total_upload_bytes = uploaded_files_total_bytes(uploaded_files)
        estimated_seconds = estimate_processing_seconds(total_upload_bytes)

        st.markdown(
            pipeline_progress_html(
                ["Extract", "Chunk", "Embed", "Compare", "Report"],
                active_index=-1,
                estimated_seconds=estimated_seconds,
            ),
            unsafe_allow_html=True,
        )
    # 2. GOOGLE DRIVE IMPORT SECTION

    if uploaded_files:
        username = st.session_state.get("username", "anonymous")
        if is_upload_rate_limited(username):
            current_count = get_upload_count(username)
            st.error(f"🚨 Upload rate limit exceeded. Current: {current_count}/100.")
            uploaded_files = None
        else:
            for _ in uploaded_files:
                increment_upload_count(username)

    # CSV Column Configuration Section
    csv_configs = {}
    csv_files = (
        [f for f in uploaded_files if f.name.lower().endswith(".csv")]
        if uploaded_files
        else []
    )
    if csv_files:
        st.markdown("### 📊 CSV Ingestion Settings")
        for f in csv_files:
            try:
                csv_bytes = f.getvalue()
                df = pd.read_csv(_io.BytesIO(csv_bytes))
                columns = list(df.columns)
                if not columns:
                    st.error(f"⚠️ CSV file '{f.name}' has no columns.")
                    continue
                # Auto-detect default text column
                default_text_idx = 0
                for i, col in enumerate(columns):
                    if any(
                        term in col.lower()
                        for term in [
                            "response",
                            "answer",
                            "text",
                            "essay",
                            "content",
                            "document",
                            "submission",
                        ]
                    ):
                        default_text_idx = i
                        break
                # Auto-detect default name/id column
                default_name_idx = None
                for i, col in enumerate(columns):
                    if (
                        any(
                            term in col.lower()
                            for term in [
                                "name",
                                "student",
                                "email",
                                "id",
                                "user",
                                "username",
                                "timestamp",
                            ]
                        )
                        and i != default_text_idx
                    ):
                        default_name_idx = i
                        break
                st.markdown(f"**Column Mapping for `{f.name}`**")
                col_text, col_name = st.columns(2)
                with col_text:
                    text_col = st.selectbox(
                        f"Text Column ({f.name})",
                        options=columns,
                        index=default_text_idx,
                        key=f"csv_text_col_{f.name}",
                        help="Select the column containing the essay/text responses to analyze.",
                    )
                with col_name:
                    name_options = ["None (Use Row Number)"] + columns
                    default_name_idx_adjusted = (
                        (default_name_idx + 1) if default_name_idx is not None else 0
                    )
                    name_col = st.selectbox(
                        f"Student Name/ID Column ({f.name})",
                        options=name_options,
                        index=default_name_idx_adjusted,
                        key=f"csv_name_col_{f.name}",
                        help="Select the column containing student names or IDs (optional).",
                    )
                csv_configs[f.name] = {
                    "df": df,
                    "text_col": text_col,
                    "name_col": (
                        None if name_col == "None (Use Row Number)" else name_col
                    ),
                }
            except (ValueError, OSError, TypeError, KeyError) as e:
                st.error(f"❌ Failed to parse CSV file '{f.name}': {str(e)}")

    st.markdown("### 🔗 Or Upload via Public URL")

    # Initialise URL-related session state keys so they persist across reruns
    if "url_text" not in st.session_state:
        st.session_state.url_text = None
    if "url_filename" not in st.session_state:
        st.session_state.url_filename = None
    if "_last_fetched_url" not in st.session_state:
        st.session_state._last_fetched_url = ""

    _url_col, _btn_col = st.columns([5, 1])
    with _url_col:
        url_input = st.text_input(
            "Paste a direct URL to a document or webpage",
            placeholder="https://example.com/paper.pdf",
            key="url_input",
            help='Enter a public URL to a PDF, DOCX, DOC, TXT file, or webpage. Click "Fetch" to load it.',
        )
    with _btn_col:
        st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
        fetch_url_btn = st.button(
            "Fetch", key="fetch_url_btn", use_container_width=True
        )

    # 2. GOOGLE DRIVE IMPORT SECTION (#146)

    from src.utils.google_drive import bulk_download_drive_folder

    # Clear cached URL result when the user changes the URL field
    if url_input.strip() != st.session_state._last_fetched_url:
        if st.session_state.url_text is not None:
            st.session_state.url_text = None
            st.session_state.url_filename = None

    # Fetch only when the button is explicitly clicked
    if fetch_url_btn and url_input and url_input.strip():
        try:
            from src.core.document_parser import extract_text_from_url

            with st.spinner("🔍 Fetching and extracting text from URL..."):
                _fetched_text = extract_text_from_url(url_input.strip())
                if not _fetched_text or len(_fetched_text.strip()) < 50:
                    st.warning(
                        "⚠️ The URL did not return enough text content for analysis."
                    )
                else:
                    from urllib.parse import urlparse as _urlparse

                    _parsed = _urlparse(url_input.strip())
                    st.session_state.url_text = _fetched_text
                    st.session_state.url_filename = sanitize_filename(
                        f"webpage_{_parsed.netloc.replace('.', '_')}.txt"
                    )
                    st.session_state._last_fetched_url = url_input.strip()
                    st.success(
                        f"✅ Successfully extracted {len(_fetched_text)} characters from the URL."
                    )
        # Requires generic catch because extract_text_from_url explicitly raises generic Exception
        except Exception as _e:
            st.error(f"❌ Failed to fetch URL: {str(_e)}")
            st.session_state.url_text = None
            st.session_state.url_filename = None

    # Show status of currently loaded URL document
    if st.session_state.url_text is not None:
        st.info(
            f"🔗 URL document loaded: **{st.session_state.url_filename}** ({len(st.session_state.url_text)} characters)"
        )

    file_bytes_dict = {}
    if uploaded_files:
        for uploaded_file in uploaded_files:
            safe_name = unique_filename(
                uploaded_file.name,
                file_bytes_dict,
            )
            file_bytes_dict[safe_name] = uploaded_file.getvalue()


# -----------------------------------------------------------------------------
# Authentication Guard
# -----------------------------------------------------------------------------
if not st.session_state.authenticated:
    st.header("🔑 Login")
    username_input = st.text_input("Username")
    password_input = st.text_input("Password", type="password")

    if st.button("Login"):
        if authenticate_user(username_input, password_input):
            st.session_state.authenticated = True
            st.session_state.username = username_input
            st.rerun()
        else:
            st.error("Invalid username or password.")
    st.stop()
    # 2. GOOGLE DRIVE IMPORT SECTION (#146)
    try:
        from src.utils.google_drive import bulk_download_drive_folder
    except ImportError:
        bulk_download_drive_folder = None

    if "drive_files_dict" not in st.session_state:
        st.session_state.drive_files_dict = {}

    if bulk_download_drive_folder is not None:
        with st.expander("🌐 Import from Google Drive Folder", expanded=False):
            drive_folder_input = st.text_input(
                "Google Drive Folder Link / ID:", key="drive_folder_url_input"
            )
            drive_api_key = st.text_input(
                "API Key (Optional):", type="password", key="drive_api_key_input"
            )

            if st.button(
                "📥 Import Files from Drive", type="primary", use_container_width=True
            ):
                if not drive_folder_input.strip():
                    st.error("🚨 Please enter a valid Google Drive folder link or ID.")
                else:
                    with st.spinner(
                        "Connecting to Google Drive API & downloading files..."
                    ):
                        try:
                            downloaded_dict, downloaded_names = (
                                bulk_download_drive_folder(
                                    folder_url_or_id=drive_folder_input,
                                    api_key=(
                                        drive_api_key.strip() if drive_api_key else None
                                    ),
                                )
                            )

                            if downloaded_dict:
                                scrubbed_drive = {
                                    n: strip_exif_metadata(d, n)
                                    for n, d in downloaded_dict.items()
                                }
                                st.session_state.drive_files_dict.update(scrubbed_drive)
                                st.success(
                                    f"✅ Imported {len(downloaded_names)} files: {', '.join(downloaded_names)}"
                                )
                                st.rerun()
                            else:
                                st.warning(
                                    "No supported files found in this Drive folder."
                                )
                        except (RuntimeError, OSError, ValueError, ImportError) as err:
                            st.error(
                                f"🚨 Failed to import from Google Drive: {str(err)}"
                            )

    # 3. MERGE LOCAL AND DRIVE FILE BYTES & ENFORCE 10MB FILE SIZE LIMIT (#169)
    MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10MB limit
    file_bytes_dict = {}
    if uploaded_files:
        # Re-initialize to handle zip/csv extraction correctly instead of raw bytes
        file_bytes_dict = {}
        for uploaded_file in uploaded_files:
            original_name = uploaded_file.name
            safe_name = unique_filename(
                original_name,
                file_bytes_dict,
            )

            if uploaded_file.size > MAX_FILE_SIZE_BYTES:
                st.error(
                    f"⚠️ File **'{safe_name}'** exceeds the maximum size "
                    f"limit of 10MB "
                    f"({uploaded_file.size / (1024 * 1024):.2f}MB). "
                    "Please upload a smaller file."
                )
                continue

            if original_name.lower().endswith(".zip"):
                try:
                    from src.utils.zip_processor import process_zip_file

                    zip_files = process_zip_file(uploaded_file.read())
                    if not zip_files:
                        st.error(
                            f"⚠️ ZIP file '{safe_name}' contains no supported documents (.pdf, .docx, .txt)."
                        )
                    else:
                        file_bytes_dict.update(
                            {
                                name: strip_exif_metadata(data, name)
                                for name, data in zip_files.items()
                            }
                        )
                except ValueError as ve:
                    st.error(
                        f"⚠️ Failed to process ZIP archive '{safe_name}': {str(ve)}"
                    )
                except (OSError, RuntimeError, TypeError):
                    st.error(
                        f"⚠️ Failed to process ZIP archive '{safe_name}': Unknown error occurred."
                    )
            elif original_name.lower().endswith(".csv"):
                if original_name in csv_configs:
                    config = csv_configs[original_name]
                    df = config["df"]
                    text_col = config["text_col"]
                    name_col = config["name_col"]
                    for idx, row in df.iterrows():
                        text_val = row[text_col]
                        if pd.isna(text_val) or not str(text_val).strip():
                            continue
                        if name_col and not pd.isna(row[name_col]):
                            student_name = str(row[name_col]).strip()
                        else:
                            student_name = f"Row {idx + 1}"
                        virtual_filename = unique_filename(
                            (f"{student_name} " f"({safe_name} - Row {idx + 1}).txt"),
                            file_bytes_dict,
                        )
                        file_bytes_dict[virtual_filename] = strip_exif_metadata(
                            str(text_val).encode("utf-8"), virtual_filename
                        )
            else:
                file_bytes_dict[safe_name] = strip_exif_metadata(
                    uploaded_file.read(),
                    safe_name,
                )
            uploaded_file.seek(0)

    # Allow analysis with existing index even without new uploads
    # Read URL result from session state (populated by the Fetch button above)
    url_text = st.session_state.url_text
    url_filename = st.session_state.url_filename

    if st.session_state.drive_files_dict:
        for drive_name, drive_bytes in st.session_state.drive_files_dict.items():
            safe_drive_name = unique_filename(
                drive_name,
                file_bytes_dict,
            )
            if len(drive_bytes) > MAX_FILE_SIZE_BYTES:
                st.error(
                    f"⚠️ Google Drive file **'{safe_drive_name}'** exceeds "
                    f"the maximum size limit of 10MB "
                    f"({len(drive_bytes) / (1024 * 1024):.2f}MB)."
                )
            else:
                file_bytes_dict[safe_drive_name] = drive_bytes

    # 3. PDF Decryption Check
    encrypted_files_detected = []
    import fitz

    for file_name, file_bytes in list(file_bytes_dict.items()):
        if file_name.lower().endswith(".pdf"):
            try:
                doc = fitz.open(stream=file_bytes, filetype="pdf")
                is_encrypted = doc.is_encrypted or doc.needs_pass
                doc.close()

                if is_encrypted:
                    user_pass = st.session_state.pdf_passwords.get(file_name, None)
                    if not user_pass:
                        encrypted_files_detected.append(file_name)
                    else:
                        doc = fitz.open(stream=file_bytes, filetype="pdf")
                        auth_success = doc.authenticate(user_pass)
                        if not auth_success:
                            encrypted_files_detected.append(file_name)
                        else:
                            file_bytes_dict[file_name] = doc.write()
                        doc.close()
            except Exception:
                pass

    if encrypted_files_detected:
        st.warning(
            "🔒 Password-protected PDF(s) detected! Please enter the password(s) below:"
        )
        for enc_file in encrypted_files_detected:
            col1, col2 = st.columns([3, 1])
            with col1:
                input_pass = st.text_input(
                    f"Password for '{enc_file}'",
                    type="password",
                    key=f"pass_input_{enc_file}",
                )
            with col2:
                st.write(" ")
                st.write(" ")
                if st.button("Decrypt PDF", key=f"btn_decrypt_{enc_file}"):
                    if input_pass:
                        st.session_state.pdf_passwords[enc_file] = input_pass
                        st.success(f"Password saved for {enc_file}!")
                        st.rerun()
                    else:
                        st.error("Please enter a password.")
        st.stop()

    # ── Display Failed Documents & Retry OCR Button (#183) ───────────────────
    if st.session_state.failed_documents:
        failed_list = list(st.session_state.failed_documents.keys())
        st.warning(
            f"⚠️ **{len(failed_list)} document(s) failed text extraction/OCR:** "
            f"`{', '.join(failed_list)}`. This can happen due to transient memory errors."
        )
        col_retry, _ = st.columns([1, 3])
        with col_retry:
            if st.button("🔄 Retry OCR", key="retry_ocr_button", type="secondary"):
                # Re-add failed document bytes to active dict and clear failed state
                for fname, fbytes in st.session_state.failed_documents.items():
                    file_bytes_dict[fname] = fbytes
                st.session_state.failed_documents = {}
                st.rerun()

    # 4. PIPELINE STOP CHECK
    if len(file_bytes_dict) < 2 and url_text is None:
        if st.session_state.analysis_results is None:
            st.markdown(
                empty_state_html(
                    "Waiting for Files",
                    "Please upload or import from Drive at least 2 PDF, DOCX, DOC, or TXT assignments (under 10MB each) to begin.",
                    "📂",
                ),
                unsafe_allow_html=True,
            )
            st.stop()

    st.markdown("### 📝 Set Document Metadata")
    col1, col2 = st.columns(2)
    with col1:
        batch_class = st.text_input("Default Class/Section", value="Class A")
    with col2:
        batch_assignment = st.text_input(
            "Default Assignment Title", value="Assignment 1"
        )

    col_tags = st.columns(1)[0]
    with col_tags:
        batch_tags = st.text_input("Tags (comma separated)", placeholder="#hw1, #draft")

    metadata_dict = {}
    for filename in file_bytes_dict.keys():
        # Check if this filename is a virtual CSV document
        is_csv_doc = False
        csv_filename_matched = None
        for csv_name in csv_configs.keys():
            if f"({csv_name} - Row " in filename:
                is_csv_doc = True
                csv_filename_matched = csv_name
                break

        if is_csv_doc:
            base_name = os.path.splitext(filename)[0]
            marker = f"({csv_filename_matched} - Row "
            marker_idx = base_name.find(marker)
            if marker_idx != -1:
                student_name = base_name[:marker_idx].strip()
            else:
                student_name = base_name

            metadata_dict[filename] = {
                "student_name": student_name,
                "class_section": batch_class.strip(),
                "assignment_title": batch_assignment.strip(),
                "tags": batch_tags.strip(),
            }
        else:
            base_name = os.path.splitext(filename)[0]
            guessed_name = base_name.replace("_", " ").replace("-", " ").title()

            with st.expander(f"📄 {filename}", expanded=False):
                student_name = st.text_input(
                    f"Student Name for {filename}",
                    value=guessed_name,
                    key=f"student_{filename}",
                )
                class_section = st.text_input(
                    f"Class/Section for {filename}",
                    value=batch_class,
                    key=f"class_{filename}",
                )
                assignment_title = st.text_input(
                    f"Assignment Title for {filename}",
                    value=batch_assignment,
                    key=f"assignment_{filename}",
                )

                metadata_dict[filename] = {
                    "student_name": student_name.strip(),
                    "class_section": class_section.strip(),
                    "assignment_title": assignment_title.strip(),
                }

    if url_filename:
        with st.expander(f"🔗 {url_filename}", expanded=True):
            student_name = st.text_input(
                f"Student Name for {url_filename}",
                value="Web Source",
                key=f"student_{url_filename}",
            )
            class_section = st.text_input(
                f"Class/Section for {url_filename}",
                value=batch_class,
                key=f"class_{url_filename}",
            )
            assignment_title = st.text_input(
                f"Assignment Title for {url_filename}",
                value=batch_assignment,
                key=f"assignment_{url_filename}",
            )
            metadata_dict[url_filename] = {
                "student_name": student_name.strip(),
                "class_section": class_section.strip(),
                "assignment_title": assignment_title.strip(),
                "tags": batch_tags.strip(),
            }

    @st.cache_data(show_spinner=False)
    def run_pipeline(
        file_bytes_dict: dict[str, bytes],
        ocr_language: str,
        ocr_dpi: int,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        existing_index=None,
        existing_registry=None,
        url_text: str = None,
        url_filename: str = None,
    ):
        raw_texts = {}

        failed_files = {}

        for name, data in file_bytes_dict.items():
            try:
                extracted = extract_text(
                    _io.BytesIO(data),
                    name,
                    ocr_language=ocr_language,
                    ocr_dpi=ocr_dpi,
                )
                if extracted and extracted.strip():
                    raw_texts[name] = extracted
                else:
                    failed_files[name] = data
            except Exception:
                failed_files[name] = data

        failed_files = []
        failure_details = []

        for name, data in file_bytes_dict.items():
            if not data:
                continue  # Skip dummy data used for existing index bypass
            try:
                raw_texts[name] = extract_text(
                    _io.BytesIO(data), name, ocr_language=ocr_language, ocr_dpi=ocr_dpi
                )
            except OCRDependencyError as exc:
                failed_files.append(name)
                failure_details.append(f"{name}: {exc}")

        if url_text and url_filename:
            raw_texts[url_filename] = url_text

        if failed_files:
            raise OCRFileBatchError(failed_files, failure_details)

        if "ignore_phrases" in globals() and ignore_phrases and ignore_phrases.strip():
            raw_texts = {
                name: remove_ignore_phrases(text, ignore_phrases)
                for name, text in raw_texts.items()
            }

        chunked_docs = chunk_documents(
            raw_texts, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        translated_chunked_docs = {}

        for doc_name, chunks in chunked_docs.items():
            translated_chunked_docs[doc_name] = []
            for chunk in chunks:
                prepared = prepare_text_for_embedding(chunk)
                translated_chunked_docs[doc_name].append(prepared["embedding_text"])

        embeddings = embed_documents(translated_chunked_docs)
        sim_df = document_similarity_matrix(embeddings)

        names = list(embeddings.keys())
        n = len(names)
        chunk_mat = np.zeros((n, n))

        for i, na in enumerate(names):
            for j, nb in enumerate(names):
                if i == j:
                    chunk_mat[i, j] = 1.0
                elif j > i:
                    ea, eb = embeddings[na], embeddings[nb]
                    score = (
                        float(np.max(cosine_similarity(ea, eb)))
                        if ea.size and eb.size
                        else 0.0
                    )
                    chunk_mat[i, j] = score
                    chunk_mat[j, i] = score

        chunk_sim_df = pd.DataFrame(chunk_mat, index=names, columns=names)

        memory = psutil.virtual_memory()
        if memory.percent >= 85:
            st.warning(
                "⚠️ High memory usage detected (>85%). Large FAISS indexes may cause system instability or out-of-memory crashes."
            )

        faiss_index, registry = build_index(embeddings, chunked_docs)
        ai_probabilities = detect_documents_ai_probability(chunked_docs)

        return (
            raw_texts,
            chunked_docs,
            embeddings,
            sim_df,
            chunk_sim_df,
            faiss_index,
            registry,
            ai_probabilities,
            failed_files,
        )

    with st.spinner("🧠 Processing files and building embeddings…"):
        analysis_results = run_pipeline(file_bytes_dict, ocr_language, ocr_dpi)

    (
        raw_texts,
        chunked_docs,
        embeddings,
        sim_df,
        chunk_sim_df,
        faiss_index,
        registry,
        ai_probabilities,
        pipeline_failed_files,
    ) = analysis_results

    if pipeline_failed_files:
        st.session_state.failed_documents.update(pipeline_failed_files)

    active_sim_df = chunk_sim_df if use_chunk_matrix else sim_df
    flags = flag_plagiarism(active_sim_df, threshold=threshold)

    # ── Summary Metrics ───────────────────────────────────────────────────────────

    has_enough_files = (len(file_bytes_dict) + (1 if url_text else 0)) >= 2

    # Run Pipeline if files uploaded
    def compute_pipeline_signature(
        file_bytes_dict: dict,
        ocr_language: str,
        ocr_dpi: int,
        chunk_size: int,
        chunk_overlap: int,
        url_text: str,
        url_filename: str,
    ) -> str:
        import hashlib

        h = hashlib.sha256()
        for name in sorted(file_bytes_dict.keys()):
            data = file_bytes_dict[name]
            h.update(name.encode("utf-8", errors="ignore"))
            h.update(data)
        h.update((ocr_language or "").encode("utf-8", errors="ignore"))
        h.update(str(ocr_dpi).encode("utf-8"))
        h.update(str(chunk_size).encode("utf-8"))
        h.update(str(chunk_overlap).encode("utf-8"))
        if url_text:
            h.update(url_text.encode("utf-8", errors="ignore"))
        if url_filename:
            h.update(url_filename.encode("utf-8", errors="ignore"))
        return h.hexdigest()

    is_calculating = False
    current_sig = None

    if (len(file_bytes_dict) > 0 and any(file_bytes_dict.values())) or url_text:
        current_sig = compute_pipeline_signature(
            file_bytes_dict=file_bytes_dict,
            ocr_language=ocr_language,
            ocr_dpi=ocr_dpi,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            url_text=url_text,
            url_filename=url_filename,
        )
        if st.session_state.get("processed_pipeline_signature") != current_sig:
            is_calculating = True

    if is_calculating:
        st.subheader(get_text("analysis_summary", lang=lang_code))

        # 1. Summary Metrics Skeleton
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.markdown(f"**{get_text('metric_docs', lang=lang_code)}**")
            st.markdown(
                f'<div class="{CLASS_SKELETON} {CLASS_SKELETON_METRIC}"></div>',
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(f"**{get_text('metric_pairs', lang=lang_code)}**")
            st.markdown(
                f'<div class="{CLASS_SKELETON} {CLASS_SKELETON_METRIC}"></div>',
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(f"**{get_text('metric_flagged', lang=lang_code)}**")
            st.markdown(
                f'<div class="{CLASS_SKELETON} {CLASS_SKELETON_METRIC}"></div>',
                unsafe_allow_html=True,
            )
        with col4:
            st.markdown(f"**{get_text('metric_faiss', lang=lang_code)}**")
            st.markdown(
                f'<div class="{CLASS_SKELETON} {CLASS_SKELETON_METRIC}"></div>',
                unsafe_allow_html=True,
            )
        with col5:
            st.markdown("**🎯 Threshold**")
            st.markdown(
                f'<div class="{CLASS_SKELETON} {CLASS_SKELETON_METRIC}"></div>',
                unsafe_allow_html=True,
            )
        st.divider()

    for flag in flags:

        # 2. Tabs Skeleton
        (
            tab_warnings,
            tab_faiss,
            tab_matrix,
            tab_heatmap,
            tab_drill,
            tab_analytics,
            tab_users,
            tab_trash,
        ) = st.tabs(
            [
                get_text("tab_warnings", lang=lang_code),
                get_text("tab_faiss", lang=lang_code),
                get_text("tab_matrix", lang=lang_code),
                get_text("tab_heatmap", lang=lang_code),
                get_text("tab_drill", lang=lang_code),
                get_text("tab_analytics", lang=lang_code),
                get_text("tab_users", lang=lang_code),
                get_text("tab_trash", lang=lang_code),
            ]
        )

        with tab_warnings:
            st.markdown("🏠 Home > Dashboard > **Warnings**")
            st.subheader(get_text("tab_warnings", lang=lang_code))
            st.markdown(
                f'<div class="{CLASS_SKELETON} {CLASS_SKELETON_TITLE}"></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="{CLASS_SKELETON} {CLASS_SKELETON_TEXT}"></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="{CLASS_SKELETON} {CLASS_SKELETON_TEXT}"></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="{CLASS_SKELETON} {CLASS_SKELETON_TEXT_SHORT}"></div>',
                unsafe_allow_html=True,
            )

        with tab_faiss:
            st.markdown("🏠 Home > Dashboard > **FAISS Chunk Search**")
            st.subheader("⚡ FAISS Chunk Search")
            st.markdown(
                f'<div class="{CLASS_SKELETON} {CLASS_SKELETON_TEXT_SHORT}" style="height: 40px; width: 100%;"></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="{CLASS_SKELETON} {CLASS_SKELETON_TEXT}" style="height: 200px;"></div>',
                unsafe_allow_html=True,
            )

        with tab_matrix:
            st.markdown("🏠 Home > Dashboard > **Similarity Matrix**")
            st.subheader("📋 Similarity Matrix")
            st.markdown(
                f'<div class="{CLASS_SKELETON} {CLASS_SKELETON_TABLE}"></div>',
                unsafe_allow_html=True,
            )

        with tab_heatmap:
            st.markdown("🏠 Home > Dashboard > **Heatmap & Network**")
            st.subheader(get_text("tab_heatmap", lang=lang_code))
            st.markdown(
                f'<div class="{CLASS_SKELETON} {CLASS_SKELETON_CHART}">Calculating similarities and generating heatmap...</div>',
                unsafe_allow_html=True,
            )
            st.divider()
            st.subheader("🕸️ Interactive Plagiarism Network")
            st.markdown(
                f'<div class="{CLASS_SKELETON} {CLASS_SKELETON_CHART}">Calculating similarities and generating network graph...</div>',
                unsafe_allow_html=True,
            )

        with tab_drill:
            st.markdown("🏠 Home > Dashboard > **Pair Drill-Down**")
            st.subheader("🔬 Pair Drill-Down")
            st.markdown(
                f'<div class="{CLASS_SKELETON} {CLASS_SKELETON_TEXT_SHORT}"></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="{CLASS_SKELETON} {CLASS_SKELETON_CHART}"></div>',
                unsafe_allow_html=True,
            )

        with tab_analytics:
            st.markdown("🏠 Home > Dashboard > **Analytics Dashboard**")
            st.subheader("📊 Plagiarism Analytics Dashboard")
            st.markdown(
                f'<div class="{CLASS_SKELETON} {CLASS_SKELETON_CHART}">Generating analytics trends...</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="{CLASS_SKELETON} {CLASS_SKELETON_CHART}">Generating top plagiarism charts...</div>',
                unsafe_allow_html=True,
            )

        with tab_users:
            st.markdown("🏠 Home > Dashboard > **User Management**")
            st.subheader("👤 User Management")
            st.markdown(
                f'<div class="{CLASS_SKELETON} {CLASS_SKELETON_TABLE}"></div>',
                unsafe_allow_html=True,
            )

        try:
            with st.spinner("🧠 Processing files and building embeddings…"):
                start_time = time.time()
                analysis_results = run_pipeline(
                    file_bytes_dict=file_bytes_dict,
                    ocr_language=ocr_language,
                    ocr_dpi=ocr_dpi,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    url_text=url_text,
                    url_filename=url_filename,
                )
                elapsed_time = time.time() - start_time
                st.session_state.analysis_results = analysis_results
                st.session_state.processed_pipeline_signature = current_sig
                st.toast(f"Successfully processed in {elapsed_time:.2f} seconds 🚀")
        except OCRFileBatchError as exc:
            from src.errors import OCR_DEPENDENCIES_MISSING

            st.error(f"🚨 {OCR_DEPENDENCIES_MISSING}")
            if exc.failed_files:
                st.warning(f"Failed files: {', '.join(exc.failed_files)}")
            st.stop()

        st.rerun()

    else:
        if st.session_state.analysis_results is not None:
            (
                raw_texts,
                chunked_docs,
                embeddings,
                sim_df,
                chunk_sim_df,
                f_idx_new,
                f_reg_new,
                ai_probabilities,
            ) = st.session_state.analysis_results
            if f_idx_new is not None:
                faiss_index = f_idx_new
            if f_reg_new:
                registry = f_reg_new

    active_sim_df = chunk_sim_df if use_chunk_matrix else sim_df
    flags = flag_plagiarism(
        active_sim_df,
        threshold=threshold,
        chunked_docs=chunked_docs,
        embeddings=embeddings,
    )

    # Network Graph Node Click Filtering setup
    selected_document_id = st.session_state.get("selected_document_id")
    if selected_document_id:
        filtered_flags = [
            flag
            for flag in flags
            if (
                flag["doc_a"] == selected_document_id
                or flag["doc_b"] == selected_document_id
            )
        ]
    else:
        filtered_flags = flags

    from src.core.tag_manager import TagManager
    from src.db.corpus_db import get_document_tags

    # Fetch tags for filtering
    active_tag = st.session_state.get("selected_tag", "All Tags")
    if active_tag != "All Tags":
        flagged_tags_filtered = []
        for flag in filtered_flags:
            tags_a = get_document_tags(flag["doc_a"])
            tags_b = get_document_tags(flag["doc_b"])
            # Include if EITHER document has the selected tag
            if TagManager.has_matching_tag(
                tags_a, active_tag
            ) or TagManager.has_matching_tag(tags_b, active_tag):
                flagged_tags_filtered.append(flag)
        filtered_flags = flagged_tags_filtered

        # Run Pipeline if files uploaded
        if (len(file_bytes_dict) > 0 and any(file_bytes_dict.values())) or url_text:
            try:
                with st.spinner("🧠 Processing files and building embeddings…"):
                    analysis_results = run_pipeline(
                        file_bytes_dict=file_bytes_dict,
                        ocr_language=ocr_language,
                        ocr_dpi=ocr_dpi,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        url_text=url_text,
                        url_filename=url_filename,
                    )
                    (
                        raw_texts,
                        chunked_docs,
                        embeddings,
                        sim_df,
                        chunk_sim_df,
                        faiss_index,
                        registry,
                        ai_probabilities,
                    ) = analysis_results
                    st.session_state.analysis_results = analysis_results
            except OCRFileBatchError as exc:
                from src.errors import OCR_DEPENDENCIES_MISSING

                st.error(f"🚨 {OCR_DEPENDENCIES_MISSING}")
                if exc.failed_files:
                    st.warning(f"Failed files: {', '.join(exc.failed_files)}")
                st.stop()

        active_sim_df = chunk_sim_df if use_chunk_matrix else sim_df
        flags = flag_plagiarism(
            active_sim_df,
            threshold=threshold,
            chunked_docs=chunked_docs,
            embeddings=embeddings,
        )

        # Network Graph Node Click Filtering setup
        selected_document_id = st.session_state.get("selected_document_id")
        if selected_document_id:
            filtered_flags = [
                flag
                for flag in flags
                if (
                    flag["doc_a"] == selected_document_id
                    or flag["doc_b"] == selected_document_id
                )
            ]
        else:
            filtered_flags = flags
    else:
        flags = []
        filtered_flags = []
        active_sim_df = None

    if has_enough_files:
        if "sent_alerts" not in st.session_state:
            st.session_state.sent_alerts = set()

        for flag in filtered_flags:
            alert_key = (flag["doc_a"], flag["doc_b"])
            if alert_key not in st.session_state.sent_alerts:
                try:
                    dispatch_plagiarism_alert(
                        doc_a=flag["doc_a"],
                        doc_b=flag["doc_b"],
                        similarity=float(flag["similarity"]),
                    )
                    st.session_state.sent_alerts.add(alert_key)
                except (ConnectionError, RuntimeError, OSError):
                    pass

        st.subheader(get_text("analysis_summary", lang=lang_code))
        doc_names = list(raw_texts.keys())
        n_docs = len(doc_names)
        total_pairs = n_docs * (n_docs - 1) // 2 if n_docs > 1 else 0
        n_flagged = len(flags)

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric(get_text("metric_docs", lang=lang_code), n_docs)
        col2.metric(get_text("metric_pairs", lang=lang_code), total_pairs)
        col3.metric(get_text("metric_flagged", lang=lang_code), n_flagged)
        col4.metric(
            get_text("metric_faiss", lang=lang_code),
            faiss_index.ntotal if faiss_index is not None else 0,
        )
        col5.metric("🎯 Threshold", f"{threshold:.0%}")
        st.divider()

    # ── Application Tabs (Translated i18n Headers) ────────────────────────────
    (
        tab_warnings,
        tab_faiss,
        tab_matrix,
        tab_heatmap,
        tab_drill,
        tab_analytics,
        tab_users,
        tab_trash,
        tab_settings,
    ) = st.tabs(
        [
            get_text("tab_warnings", lang=lang_code),
            get_text("tab_faiss", lang=lang_code),
            get_text("tab_matrix", lang=lang_code),
            get_text("tab_heatmap", lang=lang_code),
            get_text("tab_drill", lang=lang_code),
            get_text("tab_analytics", lang=lang_code),
            get_text("tab_users", lang=lang_code),
            get_text("tab_trash", lang=lang_code),
            get_text("tab_settings", lang=lang_code),
        ],
        key="main_tabs",
    )

    # ══ TAB 1: WARNINGS ═══════════════════════════════════════════════════════
    with tab_warnings:
        st.markdown("🏠 Home > Dashboard > **Warnings**")
        st.subheader(get_text("tab_warnings", lang=lang_code))

        # LMS CSV Export (Issue #305)
        st.markdown("---")
        export_col1, export_col2 = st.columns([0.8, 0.2])
        with export_col1:
            st.caption("Generate a CSV of flagged incidents for LMS grading.")
        with export_col2:
            raw_incidents = get_all_incidents_above_threshold_for_export(
                threshold=threshold
            )
            csv_data = LMSExportEngine.generate_incident_csv(raw_incidents)
            if csv_data:
                st.download_button(
                    label="📥 Export Incident Log",
                    data=csv_data,
                    file_name="plagiarism_incident_log.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            else:
                st.button(
                    "📥 Export Incident Log", disabled=True, use_container_width=True
                )
        st.markdown("---")

        if selected_document_id:
            st.info(f"Showing warnings involving: {selected_document_id}")
            if st.button("Clear Document Filter"):
                st.session_state.selected_document_id = None
                st.rerun()

            render_warning_controls(
                filtered_flags, threshold=threshold, ai_probabilities=ai_probabilities
            )

    # ══ TAB 2: FAISS ══════════════════════════════════════════════════════════
    with tab_faiss:

        st.subheader("⚡ FAISS Vector Search")

        if faiss_index is not None:
            st.info(f"Index total: {faiss_index.ntotal} vectors.")
        else:
            st.warning("FAISS index is not initialized.")

        st.markdown("🏠 Home > Dashboard > **FAISS Chunk Search**")
        st.subheader("⚡ FAISS Chunk Search")
        st.info(f"Index total: {faiss_index.ntotal if faiss_index else 0} vectors.")

        faiss_query = st.text_input(
            "Query FAISS Index:",
            placeholder="Type a text snippet to search vector index...",
            key="faiss_query_input",
        )

        if st.button("🔍 Run FAISS Search", key="run_faiss_search_btn"):
            if faiss_query.strip() and faiss_index is not None:
                try:
                    from src.core.embeddings import generate_embeddings  # type: ignore
                    from src.core.faiss_indexer import (  # type: ignore
                        search_similar_chunks,
                    )

                    q_vec = generate_embeddings([faiss_query.strip()])[0]
                    q_results = search_similar_chunks(
                        q_vec,
                        faiss_index,
                        registry,
                        top_k=faiss_top_k if "faiss_top_k" in locals() else 5,
                        threshold=threshold,
                    )
                    st.session_state.q_results = q_results
                except Exception as err:
                    st.error(f"FAISS search error: {err}")
                    st.session_state.q_results = None
            else:
                st.warning("Please enter a valid query string.")
                st.session_state.q_results = None

        if st.session_state.get("q_results") is not None:
            q_results = st.session_state.q_results
            if not q_results:
                st.info("No matching vector chunks found above threshold.")
            else:
                min_sim, max_sim = st.slider(
                    "Similarity Range:",
                    min_value=0.0,
                    max_value=1.0,
                    value=(0.0, 1.0),
                    step=0.01,
                    format="%.2f",
                    key="faiss_similarity_range",
                )

                results_df = faiss_results_dataframe(
                    q_results,
                    min_similarity=min_sim,
                    max_similarity=max_sim,
                )

                if not results_df.empty:
                    st.caption(
                        "Click a column header to sort by similarity, "
                        "target document, chunk, or rank."
                    )
                    st.dataframe(
                        results_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Rank": st.column_config.NumberColumn(
                                "Rank",
                                help="Default relevance order.",
                                format="%d",
                                width="small",
                            ),
                            "Target Document": (
                                st.column_config.TextColumn(
                                    "Target Document",
                                    help=("Document containing the " "matching chunk."),
                                    width="medium",
                                )
                            ),
                            "Chunk": st.column_config.NumberColumn(
                                "Chunk",
                                help="One-based chunk number.",
                                format="%d",
                                width="small",
                            ),
                            "Similarity Score": (
                                st.column_config.NumberColumn(
                                    "Similarity Score",
                                    help=(
                                        "Cosine similarity between "
                                        "the query and chunk."
                                    ),
                                    format="%.1f%%",
                                    width="medium",
                                )
                            ),
                            "Matching Text": (
                                st.column_config.TextColumn(
                                    "Matching Text",
                                    help="Text from the matched chunk.",
                                    width="large",
                                )
                            ),
                        },
                        key="faiss_search_results_table",
                    )
                else:
                    st.info(
                        "No matching vector chunks found within the selected similarity range."
                    )

    # ══ TAB 3: MATRIX ═════════════════════════════════════════════════════════
    with tab_matrix:
        st.markdown("🏠 Home > Dashboard > **Similarity Matrix**")
        st.subheader("📋 Similarity Matrix")
        if active_sim_df is None:
            from src.errors import UI_SIMILARITY_MATRIX_REUPLOAD

            st.info(UI_SIMILARITY_MATRIX_REUPLOAD)
        else:

            # Apply chosen colormap to matrix styling (#186)
            st.dataframe(
                active_sim_df.style.background_gradient(cmap=heatmap_cmap).format(
                    "{:.4f}"
                ),
                use_container_width=True,
            )

            def _highlight(val: Any) -> str:
                tier = severity_key(float(val))
                if tier == "high":
                    return "background-color:#ff4b4b;color:white;font-weight:bold;"
                if tier == "medium":
                    return "background-color:#ffa500;color:white;font-weight:bold;"
                return ""

            styled_df = active_sim_df.style.format("{:.4f}").map(_highlight)
            st.dataframe(styled_df, use_container_width=True)

            # Export options row
            col_csv, col_json, col_excel = st.columns(3)
            with col_csv:
                st.download_button(
                    "Download CSV",
                    active_sim_df.to_csv().encode("utf-8"),
                    "similarity_matrix.csv",
                    "text/csv",
                    use_container_width=True,
                )
            with col_json:
                json_data = export_similarity_matrix_to_json(active_sim_df).encode(
                    "utf-8"
                )
                st.download_button(
                    "⬇️ Download JSON",
                    json_data,
                    "similarity_matrix.json",
                    "application/json",
                    key="json_export_button",
                    use_container_width=True,
                )
            with col_excel:
                excel_data = export_similarity_matrix_to_excel(
                    active_sim_df, threshold=threshold
                )
                st.download_button(
                    "Download Excel",
                    excel_data,
                    "similarity_matrix_styled.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )

    # ══ TAB 4: HEATMAP & NETWORK ══════════════════════════════════════════════
    with tab_heatmap:

        st.subheader("🗺️ Similarity Heatmap")

        heatmap_fig = plot_similarity_heatmap(
            active_sim_df,
            title="Document Semantic Similarity",
            threshold=threshold,
            theme_colors=get_colors(),
            cmap=heatmap_cmap,  # Dynamic colormap support (#186)
        )
        st.pyplot(heatmap_fig, use_container_width=True)

        # ══ TAB 5: PAIR DRILL-DOWN ══════════════════════════════════════════════════

        st.markdown("🏠 Home > Dashboard > **Heatmap & Network**")
        st.subheader(get_text("tab_heatmap", lang=lang_code))
        if not has_enough_files or active_sim_df is None:
            st.markdown(
                empty_state_html(
                    "Waiting for Files",
                    "Please upload at least 2 PDF, DOCX, DOC, or TXT assignments to begin analysis.",
                    "📂",
                ),
                unsafe_allow_html=True,
            )
        else:
            heatmap_fig = plot_similarity_heatmap(
                active_sim_df,
                title="Document Semantic Similarity",
                threshold=threshold,
                theme_colors=get_colors(),
            )
            st.pyplot(heatmap_fig, use_container_width=True)

            buf = _io.BytesIO()
            heatmap_fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            buf.seek(0)
            st.download_button(
                "⬇️ Download Heatmap PNG",
                buf,
                "heatmap.png",
                "image/png",
            )
            svg_buf = _io.StringIO()
            heatmap_fig.savefig(svg_buf, format="svg", bbox_inches="tight")
            buf.seek(0)
            st.download_button(
                "⬇️ Export Heatmap SVG",
                svg_buf.getvalue(),
                "heatmap.svg",
                "image/svg+xml",
            )

            st.divider()
            st.subheader("🕸️ Interactive Plagiarism Network")
            st.caption(
                "Documents are shown as nodes. Connections appear when "
                "their similarity is greater than or equal to the selected threshold."
            )

            network_fig = plot_similarity_network(
                similarity_df=active_sim_df,
                threshold=threshold,
                title="Interactive Document Plagiarism Network",
            )

            if plotly_events is not None:
                selected_points = plotly_events(
                    network_fig,
                    click_event=True,
                    hover_event=False,
                    select_event=False,
                    key="plagiarism_network",
                )

                if selected_points:
                    clicked_point = selected_points[0]

                    point_index = clicked_point.get("pointIndex")

                    if point_index is not None and 0 <= point_index < len(doc_names):
                        clicked_document_id = doc_names[point_index]

                        st.session_state.selected_document_id = clicked_document_id
            else:
                st.plotly_chart(network_fig, use_container_width=True)

            selected_document_id = st.session_state.get("selected_document_id")

            if selected_document_id:
                filtered_flags = [
                    flag
                    for flag in flags
                    if (
                        flag["doc_a"] == selected_document_id
                        or flag["doc_b"] == selected_document_id
                    )
                ]
            else:
                filtered_flags = flags

    # ── Summary Metrics ───────────────────────────────────────────────────────────

    if len(file_bytes_dict) < 2:
        st.markdown(
            empty_state_html(
                "Waiting for Files",
                "Please upload at least 2 PDF, DOCX, DOC, or TXT assignments to begin analysis.",
                "📂",
            ),
            unsafe_allow_html=True,
        )
        st.stop()

    if "sent_alerts" not in st.session_state:
        st.session_state.sent_alerts = set()

    for flag in filtered_flags:
        alert_key = (flag["doc_a"], flag["doc_b"])
        if alert_key not in st.session_state.sent_alerts:
            try:
                dispatch_plagiarism_alert(
                    doc_a=flag["doc_a"],
                    doc_b=flag["doc_b"],
                    similarity=float(flag["similarity"]),
                )
                st.session_state.sent_alerts.add(alert_key)
            except Exception as e:
                logger.error(f"Failed to send webhook alert: {e}")

    # ══ TAB 5: PAIR DRILL-DOWN ════════════════════════════════════════════════

    with tab_drill:
        st.markdown("🏠 Home > Dashboard > **Pair Drill-Down**")
        st.subheader("🔬 Pair Drill-Down")
        st.caption("Inspect chunk-level similarity between any two documents.")

        if "expand_all_drill" not in st.session_state:
            st.session_state.expand_all_drill = False
        expand_all_drill = st.toggle(
            "Expand All",
            value=st.session_state.expand_all_drill,
            key="toggle_expand_all_drill",
        )
        st.session_state.expand_all_drill = expand_all_drill

        if not has_enough_files or active_sim_df is None:
            st.markdown(
                empty_state_html(
                    "Waiting for Files",
                    "Please upload at least 2 PDF, DOCX, DOC, or TXT assignments to begin analysis.",
                    "📂",
                ),
                unsafe_allow_html=True,
            )
        elif len(active_sim_df) < 2:
            from src.errors import UI_NEED_MIN_DOCUMENTS

            st.warning(UI_NEED_MIN_DOCUMENTS)
        else:
            c1, c2 = st.columns(2)
            with c1:
                doc_a = st.selectbox("Document A", doc_names, index=0, key="da")
            with c2:
                doc_b = st.selectbox(
                    "Document B",
                    [d for d in doc_names if d != doc_a],
                    index=0,
                    key="db",
                )

            score = float(active_sim_df.loc[doc_a, doc_b])
            st.markdown(f"**Overall Similarity:** `{score:.1%}`")
            st.progress(float(score))
            st.divider()

            drill_tab_analysis, drill_tab_viewer = st.tabs(
                ["📊 Chunk Matches & Report", "📄 Document Viewer"]
            )
            chunks_a = chunked_docs.get(doc_a, [])
            chunks_b = chunked_docs.get(doc_b, [])

            with drill_tab_analysis:
                top_pairs = find_most_similar_chunks(
                    chunks_a,
                    chunks_b,
                    embeddings[doc_a],
                    embeddings[doc_b],
                    top_k=5,
                    threshold=threshold,
                )
                for rank, (ca, cb, sim) in enumerate(top_pairs, 1):
                    is_exact = "".join(ca.split()) == "".join(cb.split())
                    badge = " :green[[Exact Match]]" if is_exact else ""
                    with st.expander(
                        f"#{rank} — {doc_a} ↔ {doc_b} — {sim:.1%}{badge}",
                        expanded=st.session_state.expand_all_drill or (rank == 1),
                    ):
                        highlighted_ca, highlighted_cb = highlight_overlap(ca, cb)
                        from src.utils.text_stats import format_text_stats

                        st.markdown(
                            f"**{doc_a} ({format_text_stats(ca)}):** {highlighted_ca}",
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f"**{doc_b} ({format_text_stats(cb)}):** {highlighted_cb}",
                            unsafe_allow_html=True,
                        )

            with drill_tab_viewer:
                selected_view_doc = st.radio(
                    "Select Document to Preview:",
                    options=[doc_a, doc_b],
                    horizontal=True,
                    key="doc_viewer_select",
                )
                doc_source = file_bytes_dict.get(selected_view_doc)
                matching_chunks_to_highlight = (
                    chunks_a if selected_view_doc == doc_a else chunks_b
                )

                if doc_source and str(selected_view_doc).lower().endswith(".pdf"):
                    with st.spinner("Generating highlighted PDF preview..."):
                        try:
                            highlighted_pdf_bytes = highlight_pdf_matches(
                                pdf_source=doc_source,
                                matching_chunks=matching_chunks_to_highlight,
                            )
                            base64_pdf = base64.b64encode(highlighted_pdf_bytes).decode(
                                "utf-8"
                            )
                            pdf_display = f"""<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="850px" type="application/pdf"></iframe>"""
                            st.markdown(pdf_display, unsafe_allow_html=True)
                        except (ValueError, RuntimeError, OSError, TypeError) as err:
                            st.error(f"🚨 Unable to render PDF preview: {str(err)}")
                else:
                    st.info("PDF Preview is only available for uploaded `.pdf` files.")

    # ══ TAB 6: Analytics ═════════════════════════════════════════════════════════

    with tab_analytics:
        st.markdown("🏠 Home > Dashboard > **Analytics Dashboard**")
        st.subheader("📊 Plagiarism Analytics Dashboard")

        # ── Document Word Counts Graph ──
        st.subheader(get_text("document_word_counts", lang=lang_code))
        word_counts = get_document_word_counts()
        if word_counts:
            word_counts_fig = plot_document_sizes(word_counts)
            st.plotly_chart(word_counts_fig, use_container_width=True)
        else:
            st.info(get_text("no_documents_db", lang=lang_code))
        st.divider()

        if not has_enough_files:
            st.markdown(
                empty_state_html(
                    "Waiting for Files",
                    "Please upload at least 2 PDF, DOCX, DOC, or TXT assignments to begin analysis.",
                    "📂",
                ),
                unsafe_allow_html=True,
            )
        else:
            if flags:
                sync_flagged_incidents(flags, threshold=threshold)
                from src.utils.bulk_export import generate_bulk_reports_zip

                analysis_results = st.session_state.get("analysis_results")
                chunked_docs = analysis_results[1] if analysis_results else None
                embeddings = analysis_results[2] if analysis_results else None

                selected_warnings = st.session_state.get("selected_warnings", set())
                export_flags = [
                    f
                    for f in flags
                    if f"{f['doc_a']}_{f['doc_b']}" in selected_warnings
                ]

                if not export_flags:
                    st.info(
                        "No warnings selected for export. Please select warnings in the Flagged Incidents tab."
                    )
                else:
                    zip_bytes = generate_bulk_reports_zip(
                        export_flags,
                        chunked_docs=chunked_docs,
                        embeddings=embeddings,
                    )
                    st.download_button(
                        label=f"⬇️ Download {len(export_flags)} Selected Flagged Pairs (ZIP)",
                        data=zip_bytes,
                        file_name="flagged_pairs_reports.zip",
                        mime="application/zip",
                        use_container_width=True,
                    )

            st.subheader("📈 High Severity Plagiarism Trends (Last 30 Days)")
            trend_data = get_high_severity_trends(days=30)
            trend_fig = plot_high_severity_trends(trend_data)
            st.plotly_chart(trend_fig, use_container_width=True)

            st.divider()
            st.subheader("🔝 Most Frequently Plagiarized Documents")
            doc_data = get_most_plagiarized_documents(limit=10)
            doc_fig = plot_most_plagiarized_documents(doc_data)
            st.plotly_chart(doc_fig, use_container_width=True)

            st.divider()

            st.subheader("📊 Similarity Score Distribution")
            analysis_results = st.session_state.get("analysis_results")
            if analysis_results is not None:
                sim_matrix = (
                    analysis_results[4] if use_chunk_matrix else analysis_results[3]
                )
                dist_fig = plot_similarity_distribution(sim_matrix)
                st.plotly_chart(dist_fig, use_container_width=True)
            else:
                st.info(
                    "Run a plagiarism analysis to see the similarity score distribution."
                )

            st.divider()

            # Summary statistics
            st.subheader("📋 Analytics Summary")
            if trend_data:
                total_high_severity = sum(item["count"] for item in trend_data)
                st.metric(
                    "Total High Severity Incidents (30 days)", total_high_severity
                )
            else:
                st.info("No high severity incidents recorded in the last 30 days.")

            if doc_data:
                st.metric(
                    "Most Plagiarized Document",
                    f"{doc_data[0]['document_name']} ({doc_data[0]['incident_count']} incidents)",
                )
            else:
                st.info("No plagiarism incidents recorded.")

    # ══ TAB 7: User Management ═══════════════════════════════════════════════════
    with tab_users:
        st.markdown("🏠 Home > Dashboard > **User Management**")
        st.subheader("👥 User Management")
        if user_role == "admin":
            users = get_all_users()
            if users:
                # Table Header
                col_h1, col_h2, col_h3, col_h4 = st.columns([2, 1, 1, 1])
                with col_h1:
                    st.markdown("**Username**")
                with col_h2:
                    st.markdown("**Role**")
                with col_h3:
                    st.markdown("**Status**")
                with col_h4:
                    st.markdown("**Actions**")
                st.write("---")

                # Table Rows
                for u in users:
                    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
                    with c1:
                        st.write(u["username"])
                    with c2:
                        st.write(u["role"].capitalize())
                    with c3:
                        is_active = u.get("is_active", True)
                        if is_active:
                            st.markdown(
                                "<span style='color: #28a745; font-weight: bold;'>🟢 Active</span>",
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                "<span style='color: #dc3545; font-weight: bold;'>🔴 Suspended</span>",
                                unsafe_allow_html=True,
                            )
                    with c4:
                        if u["username"] == "admin" or u[
                            "username"
                        ] == st.session_state.get("username"):
                            st.button(
                                "Suspend",
                                key=f"suspend_btn_{u['username']}",
                                disabled=True,
                                use_container_width=True,
                            )
                        else:
                            is_active = u.get("is_active", True)
                            btn_label = "Suspend" if is_active else "Activate"
                            if st.button(
                                btn_label,
                                key=f"suspend_btn_{u['username']}",
                                use_container_width=True,
                            ):
                                set_user_active_status(u["username"], not is_active)
                                st.success(
                                    f"User '{u['username']}' updated successfully!"
                                )
                                st.rerun()
        else:
            st.warning(
                "⚠️ Access Denied: User Management is restricted to administrators."
            )

        st.write("---")
        st.subheader("🔐 Two-Factor Authentication (2FA)")

        current_user = st.session_state.get("username", "admin")
        enabled, otp_secret = get_2fa_status(current_user)

        if enabled:
            st.success(
                "✔️ Two-Factor Authentication is currently **enabled** for your account."
            )
            with st.expander("Deactivate Two-Factor Authentication", expanded=False):
                with st.form("disable_2fa_form"):
                    disable_code = st.text_input(
                        "Verification Code", max_chars=6, key="disable_2fa_code"
                    )
                    submit_disable = st.form_submit_button(
                        "Disable 2FA", use_container_width=True
                    )
                    if submit_disable:
                        import pyotp

                        totp = pyotp.TOTP(otp_secret)
                        if totp.verify(disable_code.strip()):
                            disable_2fa(current_user)
                            st.success(
                                "✅ Two-factor authentication has been disabled."
                            )
                            st.rerun()
                        else:
                            st.error(
                                "🚨 Invalid verification code. 2FA remains enabled."
                            )
        else:
            st.info(
                "🔒 Two-Factor Authentication (2FA) is currently **disabled** for your account. We highly recommend enabling it."
            )
            if not st.session_state.get("show_2fa_setup", False):
                if st.button("Setup 2FA", use_container_width=True):
                    st.session_state.show_2fa_setup = True
                    import pyotp

                    st.session_state.temp_2fa_secret = pyotp.random_base32()
                    st.rerun()
            else:
                temp_secret = st.session_state.get("temp_2fa_secret")
                if temp_secret:
                    import pyotp

                    totp = pyotp.TOTP(temp_secret)
                    provisioning_uri = totp.provisioning_uri(
                        name=current_user, issuer_name="PlagiarismDetector"
                    )

                    st.markdown("### ⚙️ Step 1: Scan this QR Code")
                    from io import BytesIO

                    import qrcode

                    qr = qrcode.QRCode(version=1, box_size=5, border=3)
                    qr.add_data(provisioning_uri)
                    qr.make(fit=True)
                    img = qr.make_image(fill_color="black", back_color="white")
                    buf = BytesIO()
                    img.save(buf, format="PNG")
                    qr_bytes = buf.getvalue()

                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.image(qr_bytes, width=250)
                    with col2:
                        st.code(f"Account: {current_user}\nSecret Key: {temp_secret}")

                    st.markdown("### ⚙️ Step 2: Verify and Enable 2FA")
                    with st.form("verify_2fa_setup_form"):
                        setup_code = st.text_input(
                            "6-digit Code", max_chars=6, key="setup_2fa_code"
                        )
                        col_btn1, col_btn2 = st.columns(2)
                        with col_btn1:
                            submit_setup = st.form_submit_button(
                                "Verify and Enable", use_container_width=True
                            )
                        with col_btn2:
                            cancel_setup = st.form_submit_button(
                                "Cancel Setup", use_container_width=True
                            )

                        if submit_setup:
                            if totp.verify(setup_code.strip()):
                                enable_2fa(current_user, temp_secret)
                                st.session_state.show_2fa_setup = False
                                if "temp_2fa_secret" in st.session_state:
                                    del st.session_state.temp_2fa_secret
                                st.success(
                                    "🎉 Two-Factor Authentication has been successfully enabled!"
                                )
                                st.rerun()
                            else:
                                st.error("🚨 Invalid verification code.")
                        if cancel_setup:
                            st.session_state.show_2fa_setup = False
                            if "temp_2fa_secret" in st.session_state:
                                del st.session_state.temp_2fa_secret
                            st.rerun()

    # ══ TAB 8: Trash ═════════════════════════════════════════════════════════════
    with tab_trash:
        st.markdown("🏠 Home > Dashboard > **Trash**")
        st.subheader(get_text("tab_trash", lang=lang_code))

        # 1. Fetch soft-deleted documents
        deleted_docs = get_deleted_documents()

        if not deleted_docs:
            st.info(get_text("trash_empty_msg", lang=lang_code))
        else:
            # 2. Empty Trash action and confirmation
            col_empty, _ = st.columns([1, 3])
            with col_empty:
                if st.button(
                    "🗑️ Empty Trash",
                    key="empty_trash_btn",
                    type="primary",
                    use_container_width=True,
                ):
                    st.session_state.show_confirm_empty_trash = True

            if st.session_state.get("show_confirm_empty_trash"):
                st.warning(
                    "⚠️ Are you sure you want to permanently delete all soft-deleted documents? This action cannot be undone."
                )
                c_confirm, c_cancel = st.columns(2)
                with c_confirm:
                    if st.button(
                        "Yes, empty trash",
                        key="confirm_empty_trash_btn",
                        type="primary",
                        use_container_width=True,
                    ):
                        empty_trash()
                        try:
                            from src.utils.redis_cache import get_cache

                            cache = get_cache()
                            if cache.is_available():
                                cache.delete("faiss:index:corpus_index")
                                cache.clear_pattern("analysis:*")
                        except Exception as e:
                            logger.error(f"Error invalidating cache: {e}")

                        if "analysis_results" in st.session_state:
                            st.session_state.analysis_results = None
                        if "processed_pipeline_signature" in st.session_state:
                            st.session_state.processed_pipeline_signature = None

                        st.session_state.show_confirm_empty_trash = False
                        st.success("Trash emptied successfully.")
                        st.rerun()
                with c_cancel:
                    if st.button(
                        "Cancel", key="cancel_empty_trash_btn", use_container_width=True
                    ):
                        st.session_state.show_confirm_empty_trash = False
                        st.rerun()

            st.markdown("---")

            # 3. Filter and pagination
            trash_filter = st.text_input(
                "Filter trash documents by filename", key="trash_mgmt_filter"
            )
            filtered_trash = [
                d
                for d in deleted_docs
                if not trash_filter
                or trash_filter.lower() in str(d["filename"]).lower()
            ]

            st.write(f"**{len(filtered_trash)}** documents matching in trash")

            trash_items_per_page = 10
            trash_total_pages = max(
                1,
                (len(filtered_trash) + trash_items_per_page - 1)
                // trash_items_per_page,
            )

            if "trash_doc_page" not in st.session_state:
                st.session_state.trash_doc_page = 1

            # Reset page if it exceeds total pages
            if st.session_state.trash_doc_page > trash_total_pages:
                st.session_state.trash_doc_page = 1

            current_trash_page = st.session_state.trash_doc_page
            start_idx = (current_trash_page - 1) * trash_items_per_page
            end_idx = min(start_idx + trash_items_per_page, len(filtered_trash))
            page_items = filtered_trash[start_idx:end_idx]

            # 4. List items
            for doc in page_items:
                col_info, col_actions = st.columns([5, 2])
                with col_info:
                    st.markdown(f"📄 **{doc['filename']}**")
                    st.caption(
                        f"Deleted at: {doc['deleted_at']} | Uploaded: {doc['upload_date']}"
                    )
                with col_actions:
                    btn_col1, btn_col2 = st.columns(2)
                    with btn_col1:
                        if st.button(
                            "🔄",
                            key=f"restore_btn_{doc['filename']}",
                            help="Restore document",
                        ):
                            st.session_state._pending_restore = doc["filename"]
                            st.rerun()
                    with btn_col2:
                        if st.button(
                            "❌",
                            key=f"perm_del_btn_{doc['filename']}",
                            help="Permanently delete",
                        ):
                            st.session_state._pending_perm_delete = doc["filename"]
                            st.rerun()

            # 5. Pagination controls
            if trash_total_pages > 1:
                st.markdown("<br>", unsafe_allow_html=True)
                p_col1, p_col2, p_col3 = st.columns([1, 2, 1])
                with p_col1:
                    if st.button(
                        "Prev",
                        disabled=(current_trash_page == 1),
                        key="prev_trash_page_btn",
                    ):
                        st.session_state.trash_doc_page = current_trash_page - 1
                        st.rerun()
                with p_col2:
                    st.markdown(
                        f"<div style='text-align: center; margin-top: 5px;'><small>Page {current_trash_page}/{trash_total_pages}</small></div>",
                        unsafe_allow_html=True,
                    )
                with p_col3:
                    if st.button(
                        "Next",
                        disabled=(current_trash_page == trash_total_pages),
                        key="next_trash_page_btn",
                    ):
                        st.session_state.trash_doc_page = current_trash_page + 1
                        st.rerun()

            # 6. Action confirmations
            pending_restore = st.session_state.get("_pending_restore")
            if pending_restore:
                st.markdown("---")
                st.warning(
                    f"Are you sure you want to restore **{pending_restore}** to the active corpus?"
                )
                col_c, col_a = st.columns(2)
                with col_c:
                    if st.button(
                        "Yes, restore",
                        key="confirm_restore_btn",
                        type="primary",
                        use_container_width=True,
                    ):
                        restore_document(pending_restore)
                        try:
                            from src.utils.redis_cache import get_cache

                            cache = get_cache()
                            if cache.is_available():
                                cache.delete("faiss:index:corpus_index")
                                cache.clear_pattern("analysis:*")
                        except Exception as e:
                            logger.error(f"Error invalidating cache: {e}")

                        if "analysis_results" in st.session_state:
                            st.session_state.analysis_results = None
                        if "processed_pipeline_signature" in st.session_state:
                            st.session_state.processed_pipeline_signature = None

                        embeddings_matrix = get_all_embeddings()
                        if embeddings_matrix.size > 0:
                            new_index = build_index_from_matrix(embeddings_matrix)
                            save_index(new_index, _INDEX_PATH)
                        else:
                            if os.path.exists(_INDEX_PATH):
                                os.remove(_INDEX_PATH)
                        del st.session_state._pending_restore
                        st.success(f"Successfully restored {pending_restore}.")
                        st.rerun()
                with col_a:
                    if st.button(
                        "Cancel", key="cancel_restore_btn", use_container_width=True
                    ):
                        del st.session_state._pending_restore
                        st.rerun()

            pending_perm_delete = st.session_state.get("_pending_perm_delete")
            if pending_perm_delete:
                st.markdown("---")
                st.warning(
                    f"⚠️ Are you sure you want to PERMANENTLY delete **{pending_perm_delete}**? Chunks and embeddings will be lost forever."
                )
                col_c, col_a = st.columns(2)
                with col_c:
                    if st.button(
                        "Yes, delete permanently",
                        key="confirm_perm_delete_btn",
                        type="primary",
                        use_container_width=True,
                    ):
                        permanently_delete_document(pending_perm_delete)
                        try:
                            from src.utils.redis_cache import get_cache

                            cache = get_cache()
                            if cache.is_available():
                                cache.delete("faiss:index:corpus_index")
                                cache.clear_pattern("analysis:*")
                        except Exception as e:
                            logger.error(f"Error invalidating cache: {e}")

                        if "analysis_results" in st.session_state:
                            st.session_state.analysis_results = None
                        if "processed_pipeline_signature" in st.session_state:
                            st.session_state.processed_pipeline_signature = None

                        del st.session_state._pending_perm_delete
                        st.success(f"Permanently deleted {pending_perm_delete}.")
                        st.rerun()
                with col_a:
                    if st.button(
                        "Cancel", key="cancel_perm_del_btn", use_container_width=True
                    ):
                        del st.session_state._pending_perm_delete
                        st.rerun()

    # ══ TAB 9: Settings ══════════════════════════════════════════════════════════
    with tab_settings:
        st.markdown("🏠 Home > Dashboard > **Settings**")
        st.subheader(get_text("settings", lang=lang_code))

        selected_lang_name = st.selectbox(
            "🌐 Language / Idioma",
            options=list(_SUPPORTED_LANGUAGES.values()),
            index=0,
            key="lang_selector",
        )
        lang_code = "es" if selected_lang_name == "Español" else "en"

        selected_theme = st.radio(
            get_text("theme", lang=lang_code),
            options=["Light", "Dark"],
            index=0 if current_theme == "Light" else 1,
            horizontal=True,
            key="theme_selector",
            on_change=save_preferences_callback,
        )
        if selected_theme != current_theme:
            set_theme(selected_theme)
            st.rerun()

        st.markdown("---")
        st.markdown("### 🔒 Privacy")
        st.toggle(
            "📊 Share Anonymous Usage Data",
            value=st.session_state.get("telemetry_opt_in", True),
            help=(
                "When enabled, anonymous usage metrics (such as total user and "
                "document counts) help us improve the app. You can opt out at "
                "any time — this only affects your own account."
            ),
            key="telemetry_opt_in_toggle",
            on_change=save_preferences_callback,
        )

        if user_role == "admin":
            st.markdown("---")
            st.markdown("### ⚙️ Advanced Configuration")

            threshold = st.slider(
                get_text("threshold", lang=lang_code),
                min_value=0.0,
                max_value=1.0,
                value=DEFAULT_THRESHOLDS.plagiarism,
                step=0.01,
                help=(
                    "Controls which pairs are flagged. Severity remains Medium "
                    f"at {DEFAULT_THRESHOLDS.medium:.0%} and High "
                    f"at {DEFAULT_THRESHOLDS.high:.0%}."
                ),
                key="threshold_slider",
                on_change=save_preferences_callback,
            )
            st.query_params["threshold"] = f"{threshold:.2f}"
            st.session_state.last_seen_threshold_query = f"{threshold:.2f}"

            selected_class = st.selectbox(
                "Filter by Class Section",
                options=unique_classes,
                key="class_filter_selectbox",
            )

            use_chunk_matrix = st.checkbox(
                "Use chunk-level similarity matrix",
                value=False,
                key="chunk_matrix_checkbox",
            )

            faiss_top_k = st.slider(
                "FAISS: matches per chunk",
                1,
                20,
                value=5,
                key="faiss_top_k_slider",
            )

            with st.expander("✂️ Ignore Phrases", expanded=False):
                st.caption(
                    "Enter common template text or standard assignment questions to ignore during analysis. "
                    "These phrases will be removed from documents before chunking and embedding."
                )
                ignore_phrases = st.text_area(
                    "Ignore Phrases (one per line)",
                    placeholder="Q1: Explain the theory of relativity\nQ2: Describe the process of photosynthesis",
                    help="Each line represents a phrase to ignore.",
                    key="ignore_phrases_textarea",
                )

            st.markdown("### ✂️ Chunking Settings")
            chunk_size = st.slider(
                "Chunk Size (characters)",
                200,
                2000,
                value=500,
                step=50,
                help="Target character length for text chunks during embedding.",
                key="chunk_size_slider",
            )
            chunk_overlap = st.slider(
                "Chunk Overlap (characters)",
                0,
                500,
                value=50,
                step=10,
                help="Character overlap between consecutive chunks to preserve contextual boundary.",
                key="chunk_overlap_slider",
            )

            ocr_language = DEFAULT_OCR_LANGUAGE
            ocr_dpi = DEFAULT_OCR_DPI

            with st.expander("🔤 OCR Settings", expanded=False):
                st.caption(
                    "Used only for scanned or image-only PDF pages. Text-based PDFs continue to use native extraction."
                )
                ocr_language_labels = {
                    display_name: code
                    for code, display_name in SUPPORTED_OCR_LANGUAGES.items()
                }
                language_names = list(ocr_language_labels)
                default_language_name = SUPPORTED_OCR_LANGUAGES[DEFAULT_OCR_LANGUAGE]

                selected_ocr_language_name = st.selectbox(
                    "OCR Language",
                    options=language_names,
                    index=language_names.index(default_language_name),
                    key="ocr_language_selector",
                )
                ocr_language = ocr_language_labels[selected_ocr_language_name]

                ocr_dpi = st.slider(
                    "OCR DPI Resolution",
                    min_value=150,
                    max_value=400,
                    value=DEFAULT_OCR_DPI,
                    step=25,
                    key="ocr_dpi_slider",
                )

            st.markdown("")
            if st.button(
                "🔄 Reset to Factory Defaults",
                key="reset_defaults_button",
                use_container_width=True,
            ):
                keys_to_reset = [
                    "theme_selector",
                    "threshold_slider",
                    "class_filter_selectbox",
                    "chunk_matrix_checkbox",
                    "faiss_top_k_slider",
                    "ignore_phrases_textarea",
                    "chunk_size_slider",
                    "chunk_overlap_slider",
                    "ocr_language_selector",
                    "ocr_dpi_slider",
                ]
                for key in keys_to_reset:
                    if key in st.session_state:
                        del st.session_state[key]
                if "threshold" in st.query_params:
                    del st.query_params["threshold"]
                set_theme("Light")
                st.success("✅ Settings reset to defaults!")

            st.markdown("")
            if st.button(
                "🛡️ Check Database Integrity",
                key="check_db_integrity_button",
                use_container_width=True,
            ):
                from src.db.corpus_db import check_database_integrity

                results = check_database_integrity()
                if results and all(r.lower() == "ok" for r in results):
                    st.toast("✅ Database integrity check passed (healthy).")
                else:
                    st.error(f"🚨 Database integrity check failed: {results}")
                st.rerun()

            st.markdown("---")
            st.markdown("### 🗄️ Database Backup")
            st.caption(
                "Download a consistent snapshot of the corpus SQLite "
                "database for debugging or offline backup."
            )

            try:
                corpus_database_snapshot = create_corpus_database_snapshot()
            except (OSError, sqlite3.DatabaseError) as exc:
                st.warning(
                    "The corpus database is currently unavailable for "
                    f"download: {exc}"
                )
            else:
                st.download_button(
                    label="⬇️ Download raw Database",
                    data=corpus_database_snapshot,
                    file_name="corpus.db",
                    mime="application/vnd.sqlite3",
                    key="download_raw_corpus_database",
                    use_container_width=True,
                    help=(
                        "Downloads a transactionally consistent SQLite "
                        "snapshot. Existing server data is not modified."
                    ),
                )

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
                st.rerun()

            st.markdown("")
            if st.button(
                "🧹 Clear Telemetry Cache",
                key="clear_telemetry_button",
                use_container_width=True,
                help="Flushes all cached telemetry metrics (user count, document count) from Redis.",
            ):
                TelemetryService.clear_telemetry_data()
                st.success("✅ Telemetry cache cleared successfully!")
                st.rerun()

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()

st.caption(f"🎓 {APP_TITLE} · Streamlit")

# ── Version / Update indicator ────────────────────────────────────────────────
# Import here (deferred) to avoid slowing down the initial module load for
# users who never reach the footer.
from src.utils.version_check import APP_VERSION, check_for_update_sync  # noqa: E402


@st.cache_data(ttl=3600)
def _cached_version_check() -> str | None:
    """Check for updates once per hour, cached by Streamlit."""
    return check_for_update_sync(APP_VERSION)


_latest_tag: str | None = _cached_version_check()

_footer_col1, _footer_col2 = st.columns([3, 1])
with _footer_col1:
    st.caption(
        f"🎓 {APP_TITLE} · v{APP_VERSION} · Streamlit · [🐛 Report Bug / Feedback](https://github.com/Ganesh-403/semantic-plagiarism-detector/issues)"
    )
with _footer_col2:
    if _latest_tag:
        st.markdown(
            version_check_widget_html(
                local_version=APP_VERSION,
                latest_tag=_latest_tag,
            ),
            unsafe_allow_html=True,
        )
    else:
        st.caption("✅ Up to date")
