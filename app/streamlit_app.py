import asyncio
import io as _io
import logging
import os
import traceback
import functools
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import psutil
import streamlit as st

# 1. Fix Streamlit import paths FIRST so 'app' can be found
FILE_PATH = Path(__file__).resolve()
ROOT_DIR = FILE_PATH.parent.parent  # Points to semantic-plagiarism-detector/
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# 2. Now import centralized session state keys safely
from app.session_keys import SessionKeys

# Silence harmless Windows asyncio Proactor connection lost bugs
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import json

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Standard / Third-party imports
from src.utils.temp_manager import purge_expired_temp_files

from dotenv import load_dotenv

load_dotenv()

from src.security.metadata_stripper import strip_exif_metadata
from src.utils.filename import (
    InvalidFileExtensionError,
    sanitize_filename,
    unique_filename,
    validate_document_extension,
    compute_file_hash_stream,
)


try:
    from streamlit_plotly_events import plotly_events  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    plotly_events = None

from src.core.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


class ChunkRecord:
    def __init__(self, doc_name, chunk_index, chunk_text, chunk_id=None):
        self.doc_name = doc_name
        self.chunk_index = chunk_index
        self.chunk_text = chunk_text
        self.chunk_id = chunk_id


def run_pipeline(file_bytes_dict, ocr_language, ocr_dpi, chunk_size, chunk_overlap):
    """Run the document parsing -> chunking -> embedding -> similarity pipeline."""
    raw_texts = []
    chunked_docs = []
    embeddings = []
    registry = []
    ai_probabilities = []

    if not file_bytes_dict:
        empty_sim_df = pd.DataFrame(columns=["doc_a", "doc_b", "similarity"])
        empty_chunk_df = pd.DataFrame(
            columns=["doc_name", "chunk_index", "chunk_text", "similarity"]
        )
        return (
            raw_texts,
            chunked_docs,
            np.empty((0, 0), dtype=float),
            empty_sim_df,
            empty_chunk_df,
            None,
            registry,
            ai_probabilities,
        )

    for filename, file_bytes in file_bytes_dict.items():
        try:
            extracted_text = extract_text(
                file_bytes,
                filename=filename,
                language=ocr_language,
                dpi=ocr_dpi,
            )
        except Exception:
            extracted_text = ""

        if not extracted_text:
            continue

        prepared_text = prepare_text_for_embedding(extracted_text)
        raw_texts.append(prepared_text)

        text_chunks = chunk_documents(
            [prepared_text],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        if not text_chunks:
            continue

        chunked_docs.extend(text_chunks)
        chunk_vectors = embed_chunks(text_chunks)
        if isinstance(chunk_vectors, np.ndarray):
            embeddings.extend(chunk_vectors.tolist())
        else:
            embeddings.extend(chunk_vectors)

        for chunk_index, chunk_text in enumerate(text_chunks):
            registry.append(
                ChunkRecord(
                    doc_name=filename,
                    chunk_index=chunk_index,
                    chunk_text=chunk_text,
                    chunk_id=f"{filename}:{chunk_index}",
                )
            )

    if embeddings:
        emb_matrix = np.asarray(embeddings, dtype=float)
        if emb_matrix.ndim == 1:
            emb_matrix = emb_matrix.reshape(1, -1)
        faiss_index = build_index_from_matrix(emb_matrix)
    else:
        emb_matrix = np.empty((0, 0), dtype=float)
        faiss_index = None

    doc_names = [Path(name).stem for name in file_bytes_dict.keys()]
    if len(raw_texts) > 1:
        doc_embeddings = []
        for text in raw_texts:
            text_chunks = chunk_documents(
                [text],
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            if not text_chunks:
                continue
            chunk_vectors = embed_chunks(text_chunks)
            if isinstance(chunk_vectors, np.ndarray):
                doc_embeddings.append(np.mean(chunk_vectors, axis=0))
            else:
                doc_embeddings.append(
                    np.mean(np.asarray(chunk_vectors, dtype=float), axis=0)
                )

        if doc_embeddings:
            doc_matrix = np.asarray(doc_embeddings, dtype=float)
            if doc_matrix.ndim == 1:
                doc_matrix = doc_matrix.reshape(1, -1)
            sim_matrix = cosine_similarity(doc_matrix)
            sim_rows = []
            for i in range(len(doc_names)):
                for j in range(i + 1, len(doc_names)):
                    sim_rows.append(
                        {
                            "doc_a": doc_names[i],
                            "doc_b": doc_names[j],
                            "similarity": float(sim_matrix[i, j]),
                        }
                    )
            sim_df = pd.DataFrame(sim_rows)
        else:
            sim_df = pd.DataFrame(columns=["doc_a", "doc_b", "similarity"])
    else:
        sim_df = pd.DataFrame(columns=["doc_a", "doc_b", "similarity"])

    chunk_sim_df = pd.DataFrame(
        columns=["doc_name", "chunk_index", "chunk_text", "similarity"]
    )

    return (
        raw_texts,
        chunked_docs,
        emb_matrix,
        sim_df,
        chunk_sim_df,
        faiss_index,
        registry,
        ai_probabilities,
    )


def ui_exception_handler(component_name: str):
    """Decorator that catches exceptions in a UI component and shows a
    friendly error message instead of a raw Streamlit traceback."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception:
                logger.error(
                    "Component '%s' failed to render:\n%s",
                    component_name,
                    traceback.format_exc(),
                )
                st.error(f"⚠️ Failed to load component: {component_name}")
                return None

        return wrapper

    return decorator


# Validate required environment variables during application startup
REQUIRED_ENV_VARS = [
    "REDIS_URL",
    "PLAGIARISM_WEBHOOK_URL",
    "API_BEARER_TOKEN",
]

missing_env_vars = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
if missing_env_vars:
    logger.warning(
        "Missing environment variables: %s. Some features may not work correctly. "
        "Please configure them in your .env file.",
        ", ".join(missing_env_vars),
    )

# ── Project Core & Utils Imports ──────────────────────────────────────────────
from app.theme import (
    back_to_top_html,
    get_chart_colors,
    get_theme_name,
    inject_css,
    set_theme,
    version_check_widget_html,
)
from src.core.ai_detector import detect_documents_ai_probability
from src.core.config import DEFAULT_THRESHOLDS, PLAGIARISM_THRESHOLD
from src.core.document_parser import (
    DEFAULT_OCR_DPI,
    DEFAULT_OCR_LANGUAGE,
    SUPPORTED_OCR_LANGUAGES,
    extract_text,
    prepare_text_for_embedding,
)
from src.core.embedding_model import embed_chunks, embed_documents
from src.core.faiss_index import (
    build_index,
    build_index_from_matrix,
    load_index,
    load_or_rebuild_index,
    save_index,
    search_similar_chunks,
)
from src.core.similarity import (
    cosine_similarity,
    document_similarity_matrix,
    flag_plagiarism,
)
from src.core.lexical_similarity import jaccard_similarity 
from src.visualization.network_graph import (
    plot_similarity_network,
)
from src.core.text_chunking import chunk_documents
from src.db import (
    clear_all_data,
    delete_document,
    get_all_documents,
    get_all_embeddings,
    get_chunk_registry,
    get_unique_class_sections,
)
from src.db.auth import (
    authenticate_user,
    get_2fa_status,
    get_all_users,
    get_distinct_audit_event_types,
    get_security_audit_log_count,
    get_security_audit_logs,
    get_tour_completed,
    get_upload_count,
    get_user_last_login,
    get_user_preferences,
    get_user_role,
    init_db,
    is_user_active,
    set_tour_completed,
    update_user_preferences,
)
from src.db.incidents import (
    init_incident_db,
    get_all_incidents,
    sync_flagged_incidents,
)
from src.utils.bulk_export import create_documents_bulk_zip_archive
from src.utils.pdf_report import highlight_pdf_matches
from src.db.corpus_db import get_total_document_count, init_corpus_db, get_document_by_hash
from src.i18n.translator import _SUPPORTED_LANGUAGES, get_text
from src.utils.processing_time import (
    estimate_processing_seconds,
    format_processing_duration,
)
from src.utils.diff_highlighter import highlight_overlap
from src.utils.redis_cache import (
    cache_session_state,
    clear_session,
    get_analysis_results,
    get_faiss_index,
    get_session_state,
)
from src.utils.storage_metrics import calculate_storage_usage
from src.visualization.heatmap import (
    plot_similarity_heatmap,
)
from src.core.config import get_branding_config

try:
    from src.utils.warning_list import render_warning_controls, render_copy_button
    from src.visualization.analytics import (
        plot_high_severity_trends,
        plot_most_plagiarized_documents,
        plot_processing_time_breakdown,
        plot_similarity_distribution,
    )
except ImportError:
    render_warning_controls = None
    render_copy_button = None
    plot_high_severity_trends = None
    plot_most_plagiarized_documents = None
    plot_processing_time_breakdown = None
    plot_similarity_distribution = None

try:
    from src.utils.pdf_highlighter import highlight_pdf_matches
except Exception:
    highlight_pdf_matches = None

try:
    from streamlit_tour import Tour
except ImportError:
    Tour = None

# ── Auto-refresh component for the Live Incident Stream (Issue #1384) ───────
try:
    from streamlit_autorefresh import st_autorefresh  # type: ignore
except ImportError:
    st_autorefresh = None
    logger.warning(
        "streamlit-autorefresh is not installed; the auto-refresh toggle "
        "on the Live Incident Stream tab will be disabled."
    )


try:
    from src.utils.google_drive import bulk_download_drive_folder
except Exception:
    bulk_download_drive_folder = None


# Initialize databases
init_corpus_db()
init_db()

# Purge stale temp files older than 2 hours on startup
purge_expired_temp_files()
# Start lightweight REST API server for /healthz endpoint in background
import threading
import uvicorn

from src.api.app import app as fastapi_app
import src.core.app_config as app_config


def update_global_activity():
    """Update the global last_activity timestamp."""
    try:
        from src.utils.redis_cache import get_cache

        cache = get_cache()
        cache.set("spd:v1:global:last_activity", time.time())
    except Exception as e:
        logger.error(f"Failed to update global activity: {e}")


# Register Streamlit user interaction (updates on every script rerun)
update_global_activity()


def _start_api_server():
    uvicorn.run(
        fastapi_app,
        host="0.0.0.0",
        port=8000,
        log_level="warning",
    )


if not getattr(app_config, "_api_server_started", False):
    app_config._api_server_started = True

    from starlette.middleware.base import BaseHTTPMiddleware

    class ActivityMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            # Ignore health probes
            if request.url.path not in ("/health", "/healthz"):
                update_global_activity()
            return await call_next(request)

    fastapi_app.add_middleware(ActivityMiddleware)
    threading.Thread(target=_start_api_server, daemon=True).start()


def get_active_sessions_count() -> int:
    """Return the number of active Streamlit sessions."""
    try:
        from src.utils.redis_cache import get_cache, get_session_state

        cache = get_cache()
        now = time.time()
        active_count = 0
        keys = []

        if cache.is_available():
            try:
                raw_keys = cache._client.keys("spd:v1:session:*:last_interaction")
                keys = [
                    k.decode("utf-8") if isinstance(k, bytes) else k for k in raw_keys
                ]
            except Exception as e:
                logger.error(f"Failed to scan Redis session keys: {e}")

        try:
            fallback_keys = [
                k
                for k in cache.fallback_cache.keys()
                if k.startswith("spd:v1:session:") and k.endswith(":last_interaction")
            ]
            for k in fallback_keys:
                if k not in keys:
                    keys.append(k)
        except Exception as e:
            logger.error(f"Failed to scan fallback cache session keys: {e}")

        for key in keys:
            try:
                parts = key.split(":")
                if len(parts) >= 4:
                    session_id = parts[3]
                    last_interaction = get_session_state(
                        session_id,
                        "last_interaction",
                    )
                    if (
                        last_interaction is not None
                        and now - last_interaction <= 15 * 60
                    ):
                        active_count += 1
            except Exception as e:
                logger.error(f"Error checking session activity for {key}: {e}")

        return active_count

    except Exception as e:
        logger.error(f"Error in get_active_sessions_count: {e}")
        return 0


def _run_backup_daemon():
    """Background loop to create backups after inactivity."""
    last_backup_time = 0.0

    try:
        from src.utils.redis_cache import get_cache

        cache = get_cache()
        cached = cache.get("spd:v1:global:last_backup_time")
        if cached is not None:
            last_backup_time = float(cached)
    except Exception:
        pass

    logger.info("Database backup daemon started.")

    while True:
        time.sleep(30)

        try:
            from src.core.app_config import get_backup_idle_timeout
            from src.utils.redis_cache import get_cache

            cache = get_cache()

            timeout = get_backup_idle_timeout()

            last_activity = cache.get("spd:v1:global:last_activity")
            if last_activity is None:
                last_activity = time.time()
                cache.set("spd:v1:global:last_activity", last_activity)

            now = time.time()
            idle = now - last_activity

            if (
                get_active_sessions_count() == 0
                and idle >= timeout
                and last_activity > last_backup_time
            ):
                from src.db.database_backup import (
                    create_corpus_database_snapshot,
                )
                from src.db.corpus_db import get_corpus_db_path

                snapshot = create_corpus_database_snapshot()

                db_path = get_corpus_db_path()
                backup_dir = db_path.parent / "backups"
                backup_dir.mkdir(parents=True, exist_ok=True)

                filename = (
                    backup_dir
                    / f"corpus_backup_{datetime.datetime.now():%Y%m%d_%H%M%S}.db"
                )

                filename.write_bytes(snapshot)

                logger.info(f"Backup created: {filename}")

                last_backup_time = now
                cache.set(
                    "spd:v1:global:last_backup_time",
                    last_backup_time,
                )

        except Exception as e:
            logger.exception(f"Backup daemon error: {e}")


if not getattr(app_config, "_backup_daemon_started", False):
    app_config._backup_daemon_started = True
    threading.Thread(
        target=_run_backup_daemon,
        daemon=True,
    ).start()

# Generate unique session ID for this Streamlit session
if SessionKeys.SESSION_ID not in st.session_state:
    import uuid

    st.session_state[SessionKeys.SESSION_ID] = str(uuid.uuid4())

SESSION_ID = st.session_state[SessionKeys.SESSION_ID]

# FAISS index location is centralized in src.core.app_config so this module,
# src/api/app.py, src/cli.py and src/utils/mock_data.py all agree on it.
# Cast to str because faiss.write_index / faiss.read_index require str paths.
from src.core.app_config import FAISS_INDEX_PATH

_INDEX_PATH = str(FAISS_INDEX_PATH)

# Load validated branding configuration
branding_config = get_branding_config()
_INDEX_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "corpus.index")
)
try:
    from streamlit_tour import Tour
except ImportError:
    Tour = None

# -----------------------------------------------------------------------------
# Page Configuration & Session State
# -----------------------------------------------------------------------------


def configure_page_meta(title: str, icon: str) -> None:
    """
    Configure Streamlit page metadata including title, favicon, and layout.
    """
    if not isinstance(title, str) or not title.strip():
        raise ValueError("Page title must be a non-empty string.")
    if not isinstance(icon, str) or not icon.strip():
        raise ValueError("Page icon must be a non-empty string.")

    st.set_page_config(
        page_title=title.strip(),
        page_icon=icon.strip(),
        layout="wide",
        initial_sidebar_state="auto",
    )


# Initialize page metadata with dynamic branding
configure_page_meta(title="Semantic Plagiarism Detector - Dashboard", icon="🔍")


def update_page_title(tab_name: str):
    """Update browser title based on active tab."""
    st.markdown(
        f"""
        <script>
            window.parent.document.title = '{tab_name} | Semantic Plagiarism Detector';
        </script>
        """,
        unsafe_allow_html=True,
    )



if SessionKeys.AUTHENTICATED not in st.session_state:
    st.session_state[SessionKeys.AUTHENTICATED] = False
if SessionKeys.USERNAME not in st.session_state:
    st.session_state[SessionKeys.USERNAME] = None
if SessionKeys.PDF_PASSWORDS not in st.session_state:
    st.session_state[SessionKeys.PDF_PASSWORDS] = {}
if SessionKeys.LANG not in st.session_state:
    st.session_state[SessionKeys.LANG] = "en"

if SessionKeys.MODEL_LOAD_TIME not in st.session_state:
    from src.core.embedding_model import EmbeddingModelManager

    with st.spinner("Initializing Vector Embedding Model..."):
        _start_time = time.perf_counter()
        EmbeddingModelManager.get_instance().get_model()
        st.session_state[SessionKeys.MODEL_LOAD_TIME] = (
            time.perf_counter() - _start_time
        )

st.markdown(back_to_top_html(), unsafe_allow_html=True)
inject_css()


def save_preferences_callback():
    """Persist settings to user DB profile when modified."""
    if st.session_state.get(SessionKeys.AUTHENTICATED) and st.session_state.get(
        SessionKeys.USERNAME
    ):
        prefs = {
            "threshold": st.session_state.get(
                SessionKeys.THRESHOLD_SLIDER, PLAGIARISM_THRESHOLD
            ),
            "theme": st.session_state.get("theme_selector", "Light"),
        }
        update_user_preferences(st.session_state[SessionKeys.USERNAME], prefs)


def build_visualization_lazily(is_enabled, build_fn):
    """Utility to lazily load heavy chart visualizations when requested."""
    if is_enabled:
        return build_fn()
    return None


# ── Issue #1383: Cosine vs Lexical Similarity Comparison Table ─────────────────
SEMANTIC_HIGH_THRESHOLD = 0.80  # vector (cosine) score considered "high"
LEXICAL_LOW_THRESHOLD = 0.30    # lexical (jaccard) score considered "low"


def render_cosine_vs_lexical_comparison_table(
    sim_df,
    raw_texts,
    *,
    semantic_threshold: float = SEMANTIC_HIGH_THRESHOLD,
    lexical_threshold: float = LEXICAL_LOW_THRESHOLD,
):
    """Render a two-column score comparison table in the results / drill-down view.

    Compares vector-embedding similarity (cosine semantic) against Jaccard
    lexical similarity side-by-side for every unique document pair, and
    highlights pairs where the cosine score is high (>= ``semantic_threshold``)
    but the Jaccard score is low (<= ``lexical_threshold``). Such pairs are
    strong indicators of paraphrasing / semantic plagiarism that pure lexical
    matching would miss.

    Parameters
    ----------
    sim_df : pandas.DataFrame
        Square cosine similarity matrix indexed/columned by document names.
    raw_texts : Dict[str, str]
        Mapping of document name → extracted text. Used to compute Jaccard.
    semantic_threshold : float, default 0.80
        Cosine score at/above which a pair is considered semantically similar.
    lexical_threshold : float, default 0.30
        Jaccard score at/below which a pair is considered lexically dissimilar.

    Returns
    -------
    pandas.DataFrame
        The styled comparison DataFrame (also rendered to the UI).
    """
    import itertools

    if sim_df is None or raw_texts is None or len(raw_texts) < 2:
        st.info(
            "Upload at least two documents to view the Cosine vs Lexical "
            "Similarity comparison table."
        )
        return None

    doc_names = list(sim_df.columns) if sim_df is not None else list(raw_texts.keys())
    rows = []
    for da, db in itertools.combinations(doc_names, 2):
        # Cosine score from the (already-computed) semantic matrix.
        try:
            cosine_score = float(sim_df.loc[da, db])
        except Exception:
            cosine_score = 0.0

        # Jaccard lexical score from the raw extracted texts.
        text_a = raw_texts.get(da, "") or ""
        text_b = raw_texts.get(db, "") or ""
        try:
            jaccard_score = float(jaccard_similarity(text_a, text_b))
        except Exception:
            jaccard_score = 0.0

        is_semantic_only = (
            cosine_score >= semantic_threshold and jaccard_score <= lexical_threshold
        )
        rows.append(
            {
                "Document A": da,
                "Document B": db,
                "Cosine (Semantic)": cosine_score,
                "Jaccard (Lexical)": jaccard_score,
                "Semantic Only": is_semantic_only,
            }
        )

    comp_df = pd.DataFrame(rows)
    if not comp_df.empty:
        st.dataframe(comp_df, use_container_width=True)
    return comp_df

from datetime import date, timedelta


def get_date_range_preset(preset: str) -> tuple[date, date]:
    """
    Calculate start and end dates based on a given preset string.
    """
    today = date.today()
    if preset == "Today":
        return today, today
    elif preset == "Last 7 Days":
        return today - timedelta(days=6), today
    elif preset == "Last 30 Days":
        return today - timedelta(days=29), today
    else:  # "All Time"
        return date(2020, 1, 1), today


# ── SESSION TIMEOUT & ROUTE PROTECTION ────────────────────────────────────────
TIMEOUT_LIMIT = 15 * 60  # 15 minutes in seconds

cached_last_interaction = get_session_state(SESSION_ID, SessionKeys.LAST_INTERACTION)
if cached_last_interaction is not None:
    last_interaction = cached_last_interaction
elif SessionKeys.LAST_INTERACTION in st.session_state:
    last_interaction = st.session_state[SessionKeys.LAST_INTERACTION]
else:
    last_interaction = None

if last_interaction and st.session_state.get(SessionKeys.AUTHENTICATED, False):
    elapsed_time = time.time() - last_interaction
    if elapsed_time > TIMEOUT_LIMIT:
        for key in [
            SessionKeys.AUTHENTICATED,
            SessionKeys.USERNAME,
            SessionKeys.ROLE,
            SessionKeys.LAST_INTERACTION,
        ]:
            if key in st.session_state:
                del st.session_state[key]
        clear_session(SESSION_ID)
        from src.errors import UI_SESSION_EXPIRED

        st.warning(UI_SESSION_EXPIRED)
        st.stop()
    else:
        st.session_state[SessionKeys.LAST_INTERACTION] = time.time()
        cache_session_state(SESSION_ID, SessionKeys.LAST_INTERACTION, time.time())

# ── Handle OAuth Callback (GitHub / Google SSO) ──────────────────────────────
if not st.session_state.get(SessionKeys.AUTHENTICATED, False):
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
                st.session_state[SessionKeys.AUTHENTICATED] = True
                st.session_state[SessionKeys.USERNAME] = _email
                st.session_state[SessionKeys.ROLE] = _role
                st.session_state[SessionKeys.LAST_INTERACTION] = time.time()
                cache_session_state(SESSION_ID, SessionKeys.AUTHENTICATED, True)
                cache_session_state(SESSION_ID, SessionKeys.USERNAME, _email)
                cache_session_state(SESSION_ID, SessionKeys.ROLE, _role)
                cache_session_state(
                    SESSION_ID, SessionKeys.LAST_INTERACTION, time.time()
                )
                st.query_params.clear()
                st.rerun()
        else:
            st.error("🚨 SSO authentication failed. Could not retrieve your email.")
            st.query_params.clear()

# Render Login UI if not authenticated
if not st.session_state.get(SessionKeys.AUTHENTICATED, False):
    if st.session_state.get(SessionKeys.PENDING_2FA, False):
        with st.form("otp_form"):
            st.subheader("🔒 Two-Factor Authentication")
            st.info("Enter the 6-digit verification token from your authenticator app.")
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
                username = st.session_state.get(SessionKeys.PENDING_USERNAME)
                enabled, otp_secret = get_2fa_status(username)
                if enabled and otp_secret:
                    import pyotp  # type: ignore

                    totp = pyotp.TOTP(otp_secret)
                    if totp.verify(otp_code.strip()):
                        role = st.session_state.get(SessionKeys.PENDING_ROLE)
                        st.session_state[SessionKeys.AUTHENTICATED] = True
                        st.session_state[SessionKeys.USERNAME] = username
                        st.session_state[SessionKeys.ROLE] = role
                        st.session_state[SessionKeys.LAST_INTERACTION] = time.time()

                        cache_session_state(SESSION_ID, SessionKeys.AUTHENTICATED, True)
                        cache_session_state(SESSION_ID, SessionKeys.USERNAME, username)
                        cache_session_state(SESSION_ID, SessionKeys.ROLE, role)
                        cache_session_state(
                            SESSION_ID, SessionKeys.LAST_INTERACTION, time.time()
                        )
                        prefs = get_user_preferences(username)
                        st.session_state.threshold = prefs.get(
                            "threshold", DEFAULT_THRESHOLDS.plagiarism
                        )
                        st.session_state.theme = prefs.get("theme", "Light")
                        set_theme(st.session_state.theme)

                        del st.session_state[SessionKeys.PENDING_2FA]
                        del st.session_state[SessionKeys.PENDING_USERNAME]
                        del st.session_state[SessionKeys.PENDING_ROLE]

                        st.success(f"✅ Welcome back, {username}!")
                        st.rerun()
                    else:
                        st.error("🚨 Invalid verification code. Please try again.")
                else:
                    st.error("🚨 2FA configuration error. Please contact admin.")

            if cancel_submitted:
                del st.session_state[SessionKeys.PENDING_2FA]
                del st.session_state[SessionKeys.PENDING_USERNAME]
                del st.session_state[SessionKeys.PENDING_ROLE]
                st.rerun()
            st.stop()

    st.header("🔑 Login")
    username_input = st.text_input("Username")
    password_input = st.text_input("Password", type="password")

    if st.button("Login"):
        if authenticate_user(username_input, password_input):
            role = get_user_role(username_input)
            enabled, _ = get_2fa_status(username_input)
            if enabled:
                st.session_state[SessionKeys.PENDING_2FA] = True
                st.session_state[SessionKeys.PENDING_USERNAME] = username_input
                st.session_state[SessionKeys.PENDING_ROLE] = role
                st.rerun()
            else:
                st.session_state[SessionKeys.AUTHENTICATED] = True
                st.session_state[SessionKeys.USERNAME] = username_input
                st.session_state[SessionKeys.ROLE] = role
                st.session_state[SessionKeys.LAST_INTERACTION] = time.time()
                cache_session_state(SESSION_ID, SessionKeys.AUTHENTICATED, True)
                cache_session_state(SESSION_ID, SessionKeys.USERNAME, username_input)
                cache_session_state(SESSION_ID, SessionKeys.ROLE, role)
                cache_session_state(
                    SESSION_ID, SessionKeys.LAST_INTERACTION, time.time()
                )
                st.rerun()
        else:
            st.error("Invalid username or password.")
    st.stop()

user_role = st.session_state.get(SessionKeys.ROLE, "user")

# ── Top-right Theme Toggle ───────────────────────────────────────────────────
current_theme = get_theme_name()
_, theme_col = st.columns([0.94, 0.06])

with theme_col:
    theme_icon = "☀️" if current_theme == "Dark" else "🌙"
    if st.button(theme_icon, key="theme_toggle"):
        new_theme = "Light" if current_theme == "Dark" else "Dark"
        set_theme(new_theme)
        st.rerun()


# ── Dialogs ───────────────────────────────────────────────────────────────────
@st.dialog("⚠️ Confirm Logout")
def logout_dialog():
    st.write("Are you sure you want to log out?")
    st.info("Your current session will be cleared.")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Cancel", use_container_width=True, key="cancel_logout"):
            st.rerun()
    with col2:
        if st.button(
            "Log Out", type="primary", use_container_width=True, key="confirm_logout"
        ):
            username = st.session_state.get(SessionKeys.USERNAME, "unknown")
            timestamp = datetime.now(timezone.utc).isoformat()
            logger.info("User '%s' logged out at %s", username, timestamp)
            for key in [
                SessionKeys.AUTHENTICATED,
                SessionKeys.USERNAME,
                SessionKeys.ROLE,
            ]:
                if key in st.session_state:
                    del st.session_state[key]
            clear_session(SESSION_ID)
            st.rerun()

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


# ── Corpus Overview Header & Quick Actions (#1242) ───────────────────────────
header_col, action_col1, action_col2 = st.columns([0.5, 0.25, 0.25])

with header_col:
    st.subheader("📚 Corpus Overview")

with action_col1:
    if st.button(
        "🔄 Refresh Corpus Data", key="refresh_corpus_btn", use_container_width=True
    ):
        keys_to_clear = [
            SessionKeys.ANALYSIS_RESULTS,
            SessionKeys.ANALYSIS_FILE_SIGNATURE,
            SessionKeys.DRIVE_FILES_DICT,
            SessionKeys.FAILED_DOCUMENTS,
            SessionKeys.WARNING_PAGE,
        ]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]

        st.cache_data.clear()
        st.toast("Corpus dataset refreshed.", icon="✅")
        st.rerun()

with action_col2:
    if st.button(
        "🗑️ Clear All Data",
        key="open_clear_dialog_btn",
        type="secondary",
        use_container_width=True,
    ):
        clear_all_dialog()  # type: ignore

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # ── Account Info (Issue: logged-in user details expander) ──────────────
    if st.session_state.get(SessionKeys.AUTHENTICATED, False):
        _current_username = st.session_state.get(SessionKeys.USERNAME) or "Unknown"
        with st.sidebar.expander(f"👤 Logged in as: {_current_username}"):
            st.markdown(f"**Username:** {_current_username}")
            st.markdown(f"**Role:** {user_role.capitalize() if user_role else 'N/A'}")
            try:
                _last_login = get_user_last_login(_current_username)
            except Exception:
                _last_login = None
            st.markdown(f"**Last Login:** {_last_login if _last_login else 'N/A'}")

    try:
        from src.db.auth import get_upload_count
        total_scans_sidebar = get_upload_count()
    except Exception as e:
        logger.error(f"Failed to query total scan count for sidebar: {e}")
        total_scans_sidebar = 0

    st.markdown(f"Total Scans Processed: {total_scans_sidebar:,}")
    st.markdown("### ⚙️ Settings")

    lang_options = list(_SUPPORTED_LANGUAGES.values())
    st.selectbox(
        "🌐 Language",
        options=lang_options,
        key=SessionKeys.LANG_SELECTOR,
    )
    selected_lang_name = st.session_state.get(SessionKeys.LANG_SELECTOR, "English")
    _lang_reverse = {v: k for k, v in _SUPPORTED_LANGUAGES.items()}
    lang_code = _lang_reverse.get(selected_lang_name, "en")

    if user_role == "admin":
        # ── Threshold Presets (Issue #1674) ───────────────────────────────────────
        st.markdown("### 🎯 Threshold Presets")
        
        # Define preset options with descriptions
        preset_options = {
            "Strict (0.80)": 0.80,
            "Balanced (0.59)": 0.59,
            "Lenient (0.45)": 0.45,
            "Custom": None,
        }
        
        # Determine current preset based on session state threshold
        current_threshold = st.session_state.get("threshold_slider", PLAGIARISM_THRESHOLD)
        current_preset = "Custom"
        for label, value in preset_options.items():
            if value is not None and abs(current_threshold - value) < 0.001:
                current_preset = label
                break
        
        selected_preset = st.radio(
            "Select Evaluation Standard:",
            options=list(preset_options.keys()),
            index=list(preset_options.keys()).index(current_preset),
            key="threshold_preset_radio",
            horizontal=True,
            help="Choose a predefined threshold standard or use the custom slider below.",
        )
        
        # Sync preset selection with slider value
        if selected_preset != "Custom" and preset_options[selected_preset] is not None:
            st.session_state["threshold_slider"] = preset_options[selected_preset]
            # Force rerun to update the slider widget if it changed via radio
            if current_preset != selected_preset:
                st.rerun()

        threshold = st.slider(
            "Plagiarism Threshold (Hybrid)",
            0.10,
            0.99,
            value=st.session_state.get("threshold_slider", PLAGIARISM_THRESHOLD),
            step=0.01,
            help=(
                "Combined Hybrid score threshold for flagging pair plagiarism. "
                "Calculated from Lexical (exact phrase overlap) and Semantic (meaning alignment) scores. "
                "Recommended Default: 0.59 (59%)."
            ),
            key="threshold_slider",
            on_change=save_preferences_callback,
        )
        
        # If user manually changes slider, reset preset to "Custom"
        if abs(threshold - preset_options.get(selected_preset, -1)) > 0.001:
            if st.session_state.get("threshold_preset_radio") != "Custom":
                st.session_state["threshold_preset_radio"] = "Custom"
                st.rerun()

        lexical_threshold = st.slider(
            "Lexical Sensitivity Threshold",
            0.10,
            1.00,
            value=0.50,
            step=0.05,
            help=(
                "Direct word-for-word and N-gram match threshold. "
                "Higher values require near-identical text phrasing to trigger alerts. "
                "Recommended Default: 0.50 (50%)."
            ),
            key=SessionKeys.LEXICAL_THRESHOLD_SLIDER,
        )

        semantic_threshold = st.slider(
            "Semantic Sensitivity Threshold",
            0.10,
            1.00,
            value=0.65,
            step=0.05,
            help=(
                "Transformer embedding vector similarity threshold measuring conceptual alignment and paraphrasing. "
                "Higher values require strong contextual similarity even if words differ. "
                "Recommended Default: 0.65 (65%)."
            ),
            key=SessionKeys.SEMANTIC_THRESHOLD_SLIDER,
        )

        use_chunk_matrix = st.checkbox(
            "Use chunk-level similarity matrix",
            value=False,
            key=SessionKeys.CHUNK_MATRIX_CHECKBOX,
        )
        faiss_top_k = st.slider(
            "FAISS: matches per chunk",
            1,
            20,
            value=5,
            key=SessionKeys.FAISS_TOP_K_SLIDER,
        )
        from app.components.faiss_results import render_faiss_metric_badge
        render_faiss_metric_badge(st.session_state.get("faiss_index", None))

        # ── FAISS Vector Index Memory Footprint Badge (Issue #1563) ────────────
        from src.core.faiss_index import format_faiss_memory_badge
        current_faiss_index = globals().get("faiss_index")
        if current_faiss_index is None and "faiss_index" in st.session_state:
            current_faiss_index = st.session_state["faiss_index"]
        faiss_badge_text = format_faiss_memory_badge(current_faiss_index)
        st.caption(f"⚡ **{faiss_badge_text}**")

        st.markdown("### ✂️ Chunking Settings")
        chunk_size = st.slider(
            "Chunk Size (characters)",
            200,
            2000,
            value=500,
            step=50,
            help="Target character length for text chunks during embedding.",
            key=SessionKeys.CHUNK_SIZE_SLIDER,
        )
        chunk_overlap = st.slider(
            "Chunk Overlap (characters)",
            0,
            500,
            value=50,
            step=10,
            help="Character overlap between consecutive chunks to preserve contextual boundary.",
            key=SessionKeys.CHUNK_OVERLAP_SLIDER,
        )

        with st.expander("🔤 OCR Settings", expanded=False):
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
                key=SessionKeys.OCR_LANGUAGE_SELECTOR,
            )
            ocr_language = ocr_language_labels[selected_ocr_language_name]

            ocr_dpi = st.slider(
                "OCR DPI Resolution",
                min_value=150,
                max_value=400,
                value=DEFAULT_OCR_DPI,
                step=25,
                key=SessionKeys.OCR_DPI_SLIDER,
            )
    else:
        threshold = PLAGIARISM_THRESHOLD
        use_chunk_matrix = False
        faiss_top_k = 5
        chunk_size = 500
        chunk_overlap = 50
        ocr_language = DEFAULT_OCR_LANGUAGE
        ocr_dpi = DEFAULT_OCR_DPI

    # ── API Quota Usage Gauge (Issue #1566) ──────────────────────────────────
    from app.components.api_quota_gauge import render_api_quota_gauge
    render_api_quota_gauge()


unique_classes = get_unique_class_sections()
selected_classes = st.multiselect(
    "Select Class/Section(s)",
    unique_classes,
    default=unique_classes,
    key=SessionKeys.CLASS_FILTER_SELECTBOX,
)
if not selected_classes:
    selected_classes = unique_classes
    if st.button(
        "🔄 Reset All Filters", key="reset_all_filters_button", use_container_width=True
    ):
        keys_to_reset = [
            SessionKeys.THRESHOLD_SLIDER,
            SessionKeys.LEXICAL_THRESHOLD_SLIDER,
            SessionKeys.SEMANTIC_THRESHOLD_SLIDER,
            SessionKeys.CHUNK_MATRIX_CHECKBOX,
            SessionKeys.FAISS_TOP_K_SLIDER,
            SessionKeys.CHUNK_SIZE_SLIDER,
            SessionKeys.CHUNK_OVERLAP_SLIDER,
            SessionKeys.OCR_LANGUAGE_SELECTOR,
            SessionKeys.OCR_DPI_SLIDER,
            SessionKeys.CLASS_FILTER_SELECTBOX,
            "heatmap_mask_threshold",
            "heatmap_show_percentages",
            "heatmap_dim_diagonal",
            "heatmap_tab_class_filter",
        ]
        for key in keys_to_reset:
            if key in st.session_state:
                del st.session_state[key]
        if "threshold" in st.query_params:
            del st.query_params["threshold"]
        st.success("✅ Filters reset to defaults!")
        st.rerun()

    # Keyboard shortcuts
    with st.expander("⌨️ Keyboard Shortcuts"):
        st.caption("• **R**: Rerun app")
        st.caption("• **C**: Clear cache")
        st.caption("• **Tab**: Navigate focus")

    # Model load time
    if SessionKeys.MODEL_LOAD_TIME in st.session_state:
        st.divider()
        st.caption(
            f"⚡ Vector Model Loaded in {st.session_state[SessionKeys.MODEL_LOAD_TIME]:.2f} seconds"
        )

    # ── System Health Widget (Issue #1246) ──────────────────────────────────────
    # Collapsible sidebar expander showing RAM usage, disk space, and DB status
    # for administrators monitoring the application health at a glance.
    # ── Real-Time Memory Consumption Monitor (Issue #1371) ──────────────────────
    with st.expander("🖥️ System Health & Memory", expanded=False):
        try:
            import os
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            rss_mb = mem_info.rss / (1024 * 1024)
            
            # Host limit estimation or default shared host limit (e.g., 2048 MB)
            host_limit_mb = 2048.0
            try:
                from src.core.app_config import get_host_memory_limit_mb
                host_limit_mb = float(get_host_memory_limit_mb())
            except Exception:
                pass

            ram_usage_percent = min(rss_mb / host_limit_mb, 1.0)
            ram_percent_val = (rss_mb / host_limit_mb) * 100

            # Determine warning state (>80% amber warning)
            if ram_percent_val >= 80:
                st.warning(f"⚠️ High RAM Usage: {rss_mb:.1f} MB / {host_limit_mb:.0f} MB ({ram_percent_val:.0f}%)")
            else:
                st.markdown(f"**RAM Usage:** {rss_mb:.1f} MB / {host_limit_mb:.0f} MB ({ram_percent_val:.0f}%)")

            st.progress(ram_usage_percent)

        except Exception as mem_err:
            st.error(f"Failed to measure process memory: {mem_err}")
            
            # Free disk space on the partition containing the project root
            disk_usage = psutil.disk_usage(str(ROOT_DIR))
            free_disk_gb = disk_usage.free / (1024**3)
            total_disk_gb = disk_usage.total / (1024**3)
            disk_usage_percent = disk_usage.percent

            if disk_usage_percent >= 90:
                disk_indicator = "🔴"
            elif disk_usage_percent >= 75:
                disk_indicator = "🟡"
            else:
                disk_indicator = "🟢"

            st.markdown(
                f"**💿 Disk Space:** {disk_indicator} {disk_usage_percent:.1f}% used"
            )
            st.caption(f"Free: {free_disk_gb:.1f} GB · Total: {total_disk_gb:.1f} GB")

            st.divider()
            st.markdown("**🗄️ Database Status**")

            try:
                from src.core.app_config import CORPUS_DB_PATH, AUTH_DB_PATH

                corpus_db_exists = CORPUS_DB_PATH.exists()
                if corpus_db_exists:
                    st.markdown("• **Corpus DB:** 🟢 Connected")
                    corpus_size_kb = CORPUS_DB_PATH.stat().st_size / 1024
                    st.caption(f"  Size: {corpus_size_kb:.1f} KB")
                else:
                    st.markdown("• **Corpus DB:** 🟡 Not initialized")
                    st.caption("  Will be created on first data upload.")

            except Exception as db_err:
                st.markdown("• **Corpus DB:** 🔴 Error")
                st.caption(f"  {db_err}")

            try:
                auth_db_exists = AUTH_DB_PATH.exists()
                if auth_db_exists:
                    st.markdown("• **Auth DB:** 🟢 Connected")
                    auth_size_kb = AUTH_DB_PATH.stat().st_size / 1024
                    st.caption(f"  Size: {auth_size_kb:.1f} KB")
                else:
                    st.markdown("• **Auth DB:** 🟡 Not initialized")
                    st.caption("  Will be created on first login.")

            except Exception as db_err:
                st.markdown("• **Auth DB:** 🔴 Error")
                st.caption(f"  {db_err}")

            st.divider()
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count(logical=True)

            if cpu_percent >= 90:
                cpu_indicator = "🔴"
            elif cpu_percent >= 70:
                cpu_indicator = "🟡"
            else:
                cpu_indicator = "🟢"

            st.markdown(
                f"**⚡ CPU Load:** {cpu_indicator} {cpu_percent:.1f}% "
                f"({cpu_count} cores)"
            )

        except ImportError:
            st.warning("⚠️ psutil not available. System health data unavailable.")
        except Exception as health_err:
            st.error(f"Failed to load system health data: {health_err}")

        st.divider()
        from app.theme import render_timezone_footer
        render_timezone_footer()

# ── Main UI ───────────────────────────────────────────────────────────────────
st.title("🔍 Semantic Plagiarism Detection System")

# ── Live Scan Statistics Metrics Header (#1508) ───────────────────────────────
try:
    from src.db.auth import get_upload_count
    from src.db.corpus_db import get_total_document_count
    from src.db.incidents import get_all_incidents, get_total_incidents_count

    total_scans = get_upload_count()
    corpus_size = get_total_document_count()
    flagged_incidents = get_total_incidents_count()

    _incidents = get_all_incidents(limit=10000)
    if _incidents:
        avg_sim = sum(inc.get("similarity_score", 0.0) for inc in _incidents) / len(
            _incidents
        )
    else:
        avg_sim = 0.0
except Exception as e:
    logger.error(f"Failed to load dashboard metrics: {e}")
    total_scans = 0
    corpus_size = 0
    flagged_incidents = 0
    avg_sim = 0.0

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Scans", f"{total_scans:,}")
with col2:
    st.metric("Avg Similarity %", f"{avg_sim * 100:.1f}%")
with col3:
    st.metric("Flagged Incidents", f"{flagged_incidents:,}")
with col4:
    st.metric("Corpus Size", f"{corpus_size:,}")

st.markdown("---")

with st.expander("ℹ️ How Semantic Plagiarism Detection Works"):
    st.markdown("""
        - **1. Upload files** — Upload the documents you want to compare.
        - **2. AI vector embeddings generated** — The documents are converted into vector embeddings for semantic comparison.
        - **3. View similarity heatmap & incident logs** — Review detected similarities through the heatmap and incident logs.
        """)
uploaded_files = st.file_uploader(
    "📂 Upload Assignments",
    type=["pdf", "docx", "txt", "md", "markdown", "mdown"],
    accept_multiple_files=True,
    key="file_uploader",
)

if user_role != "admin":
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
                    faiss_index = build_index_from_matrix(
                        embeddings_matrix, index_type="auto"
                    )

                    from src.core.embedding_model import embed_chunks

                    query_vec = embed_chunks([query_text.strip()])[0]

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
                                f"#{rank} · {anon_doc_name} (chunk #{record.chunk_index + 1}) "
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
                                    f"Similarity: {score * 100:.1f}%</span></div>",
                                    unsafe_allow_html=True,
                                )

                        st.caption(
                            "🔒 Document names are anonymized to protect student privacy."
                        )

            except Exception as e:
                from src.errors import UI_INDEX_LOAD_FAILED

                st.error(UI_INDEX_LOAD_FAILED.format(error=str(e)))
                st.info(
                    "Please ensure documents have been indexed by an administrator."
                )
else:
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
        st.markdown(
            "<span style='color:#999;font-size:0.85rem;'>○ No index loaded</span>",
            unsafe_allow_html=True,
        )
    st.markdown("---")

file_bytes_dict = (
    {uploaded_file.name: uploaded_file.getvalue() for uploaded_file in uploaded_files}
    if uploaded_files
    else {}
)

with st.spinner("🧠 Processing files and building embeddings…"):
    analysis_results = run_pipeline(
        file_bytes_dict,
        ocr_language,
        ocr_dpi,
        chunk_size,
        chunk_overlap,
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

active_sim_df = chunk_sim_df if use_chunk_matrix else sim_df
flags = flag_plagiarism(active_sim_df, threshold=threshold)

st.subheader("📊 Analysis Summary")
st.write(
    f"Processed **{len(raw_texts)}** documents with Chunk Size: `{chunk_size}` and Overlap: `{chunk_overlap}`."
)

st.markdown("---")
st.markdown("""
**How it works**
1. Upload **PDF, DOCX, TXT, or Markdown** assignment files or import from Google Drive
2. Text is extracted according to the file type
3. Text is split into **paragraph chunks**
4. Chunks are embedded with **SentenceTransformers**
5. A **FAISS index** is built over all chunk vectors
6. Pairs above threshold are flagged
""")
st.markdown("---")
st.caption("Semantic Plagiarism Detector · FAISS edition")

if user_role == "admin":
    st.markdown("---")
    st.markdown("### 📁 Document Management")
    existing_docs = get_all_documents()
    if existing_docs:
        st.write(f"**{len(existing_docs)}** documents in database")
        for doc in existing_docs:
            st.text(doc)

    safe_last_interaction = int(last_interaction or time.time())
    st.markdown(
        f"""
        <div id="session-timer" style="
            background-color: rgba(255, 165, 0, 0.1);
            border: 1px solid rgba(255, 165, 0, 0.3);
            border-radius: 8px;
            padding: 12px;
            margin-top: 16px;
            text-align: center;
            font-family: monospace;
            font-size: 14px;
            color: #ffa500;
        ">
            ⏱️ Session expires in: <span id="timer-display">15:00</span>
        </div>
        <script>
        (function() {{
            const timeoutLimit = {TIMEOUT_LIMIT};
            const lastInteraction = {safe_last_interaction};
            const display = document.getElementById('timer-display');

            function updateTimer() {{
                const now = Math.floor(Date.now() / 1000);
                const elapsed = now - lastInteraction;
                const remaining = Math.max(0, timeoutLimit - elapsed);

                if (remaining <= 0) {{
                    display.textContent = "00:00";
                    display.parentElement.style.borderColor = "#ff4b4b";
                    display.parentElement.style.color = "#ff4b4b";
                    display.parentElement.innerHTML = "⚠️ Session Expired. Reloading...";
                    setTimeout(() => window.location.reload(), 2000);
                    return;
                }}

                const minutes = Math.floor(remaining / 60);
                const seconds = remaining % 60;
                display.textContent = `${{minutes.toString().padStart(2, '0')}}:${{seconds.toString().padStart(2, '0')}}`;

                if (remaining < 60) {{
                    display.parentElement.style.borderColor = "#ff4b4b";
                    display.parentElement.style.color = "#ff4b4b";
                }}
            }}

            updateTimer();
            setInterval(updateTimer, 1000);
        }})();
        </script>
        """,
        unsafe_allow_html=True,
    )

    if user_role == "admin":
        st.markdown("---")
        st.markdown("### 💾 Storage Space Used")
        storage_info = calculate_storage_usage()
        st.metric(
            label="Total Storage Used",
            value=storage_info["formatted_total"],
            help="Combined SQLite database + FAISS index disk usage",
        )

        st.markdown("---")
        st.markdown("### 📁 Document Management & Bulk Export")
        existing_docs = get_all_documents()
        if existing_docs:
            raw_assignment_titles = sorted(
                list(
                    {
                        (
                            doc.assignment_title
                            if hasattr(doc, "assignment_title")
                            else (doc.get("assignment_title") if isinstance(doc, dict) else None)
                        )
                        for doc in existing_docs
                    }
                    - {None, ""}
                )
            )
            assignment_titles = ["All Assignments"] + raw_assignment_titles
            selected_assignment = st.selectbox(
                "Filter by Assignment",
                options=assignment_titles,
                key="corpus_assignment_filter_selectbox",
            )
            if selected_assignment != "All Assignments":
                existing_docs = [
                    doc
                    for doc in existing_docs
                    if (
                        doc.assignment_title
                        if hasattr(doc, "assignment_title")
                        else (doc.get("assignment_title") if isinstance(doc, dict) else None)
                    )
                    == selected_assignment
                ]

            st.write(f"**{len(existing_docs)}** documents in database")

            import pandas as pd
            from src.db.corpus_db import (
                get_document_char_counts,
                get_document_word_counts,
            )

            word_counts = get_document_word_counts()
            char_counts = get_document_char_counts()

            doc_rows = []
            for doc in existing_docs:
                fn = (
                    doc.filename
                    if hasattr(doc, "filename")
                    else (doc.get("filename") if isinstance(doc, dict) else str(doc))
                )
                doc_rows.append(
                    {
                        "Select": False,
                        "Filename": fn,
                        "Word Count": word_counts.get(fn, 0),
                        "Char Count": char_counts.get(fn, 0),
                    }
                )

            corpus_df = pd.DataFrame(doc_rows)

            sel_col1, sel_col2 = st.columns(2)
            with sel_col1:
                if st.button(
                    "☑️ Select All",
                    key="sidebar_select_all_corpus_btn",
                    use_container_width=True,
                ):
                    st.session_state["corpus_select_all_toggle"] = True
                    st.rerun()
            with sel_col2:
                if st.button(
                    "⬜ Clear",
                    key="sidebar_clear_corpus_btn",
                    use_container_width=True,
                ):
                    st.session_state["corpus_select_all_toggle"] = False
                    st.rerun()

            if st.session_state.get("corpus_select_all_toggle", False):
                corpus_df["Select"] = True

            edited_df = st.data_editor(
                corpus_df,
                column_config={
                    "Select": st.column_config.CheckboxColumn(
                        "Select",
                        default=False,
                        help="Select for bulk ZIP export",
                    ),
                    "Filename": st.column_config.TextColumn("Filename", disabled=True),
                    "Word Count": st.column_config.NumberColumn(
                        "Word Count", format="%d words", disabled=True
                    ),
                    "Char Count": st.column_config.NumberColumn(
                        "Char Count", format="%d chars", disabled=True
                    ),
                },
                disabled=["Filename", "Word Count", "Char Count"],
                hide_index=True,
                key="sidebar_corpus_data_editor",
                use_container_width=True,
            )

            selected_rows = edited_df[edited_df["Select"]]
            selected_filenames = selected_rows["Filename"].tolist()

            if selected_filenames:
                zip_data = create_documents_bulk_zip_archive(selected_filenames)
                st.download_button(
                    label=f"📦 Export Selected as ZIP ({len(selected_filenames)})",
                    data=zip_data,
                    file_name=f"corpus_export_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip",
                    key="sidebar_export_selected_zip_btn",
                    use_container_width=True,
                    type="primary",
                )
            else:
                st.button(
                    "📦 Export Selected as ZIP (0)",
                    disabled=True,
                    key="sidebar_export_zip_disabled_btn",
                    use_container_width=True,
                )

            st.markdown("---")
            for doc in existing_docs:
                fn = (
                    doc.filename
                    if hasattr(doc, "filename")
                    else (doc.get("filename") if isinstance(doc, dict) else str(doc))
                )
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.text(f"📄 {fn}")
                with col2:
                    if st.button("🗑️", key=f"del_{fn}"):
                        delete_document(fn)
                        embeddings_matrix = get_all_embeddings()
                        if embeddings_matrix.size > 0:
                            new_index = build_index_from_matrix(embeddings_matrix)
                            save_index(new_index, _INDEX_PATH)
                        else:
                            if os.path.exists(_INDEX_PATH):
                                os.remove(_INDEX_PATH)
                        st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button(
            "🗑️ Clear All Documents",
            key="clear_all_documents_button",
            use_container_width=True,
        ):
            clear_all_dialog()  # type: ignore
        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("---")
        if st.button("🚪 Log Out", use_container_width=True, key="logout_button"):
            logout_dialog()

# Onboarding Tour
if (
    Tour is not None
    and user_role == "admin"
    and not get_tour_completed(st.session_state[SessionKeys.USERNAME])
):
    username = st.session_state[SessionKeys.USERNAME]
    if st.button("🎯 Start Guided Tour", key="start_tour_button", type="primary"):
        st.session_state[SessionKeys.SHOW_TOUR] = True

    if st.session_state.get(SessionKeys.SHOW_TOUR, False):
        tour_steps = [
            Tour.info(
                title="👋 Welcome to the Plagiarism Detection System!",
                desc="This guided tour will walk you through the key features to help you get started.",
            ),
            Tour.bind(
                SessionKeys.THRESHOLD_SLIDER,
                title="⚙️ Plagiarism Threshold",
                desc=f"Adjust the flagging threshold. Default is {DEFAULT_THRESHOLDS.plagiarism:.0%}.",
                side="right",
            ),
            Tour.bind(
                SessionKeys.CLASS_FILTER_SELECTBOX,
                title="🔍 Class Filter",
                desc="Filter analysis results by specific class sections.",
                side="right",
            ),
            Tour.info(
                title="📊 Analysis Dashboard",
                desc="View similarity metrics, flagged pairs, and comparisons in the tabs below.",
            ),
        ]
        tour = Tour(steps=tour_steps)
        tour.start()
        if st.button("✅ Finish Tour", use_container_width=True):
            set_tour_completed(username, True)
            st.session_state[SessionKeys.SHOW_TOUR] = False
            st.success("✅ Onboarding tour completed!")
            st.rerun()

st.title(get_text("title", lang=lang_code))
st.markdown(get_text("subtitle", lang=lang_code))
st.divider()

if user_role != "admin":
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
                    faiss_index = build_index_from_matrix(
                        embeddings_matrix, index_type="auto"
                    )
                    processed_query = query_text.strip()
                    query_vec = embed_chunks([processed_query])[0]
                    results = search_similar_chunks(
                        query_vec, faiss_index, registry, top_k=5, threshold=threshold
                    )

                    if not results:
                        st.success(
                            "✅ No significant matches found in the assignment database."
                        )
                    else:
                        st.success(
                            f"Found **{len(results)}** potentially similar passages."
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
                                f"#{rank} · {anon_doc_name} (chunk #{record.chunk_index + 1}) — {score:.1%}",
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
                                    f"<div style='background:{color};color:white;padding:8px;border-radius:4px;text-align:center;'>"
                                    f"Similarity: {score * 100:.1f}%"
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )
            except Exception as e:
                st.error(f"Error loading index: {str(e)}")
else:
    cached_index_data = get_faiss_index("corpus_index")

st.title("🔍 Semantic Plagiarism Detection System")
st.markdown(
    "Upload student PDF, DOCX, TXT, or Markdown files. Detects **semantic similarity** "
    "using transformer embeddings + **FAISS vector search**."
)
st.divider()

if user_role == "admin":
    cached_index_data = get_faiss_index("corpus_index")

    if cached_index_data is not None and os.path.exists(_INDEX_PATH):
        try:
            import faiss

            index_buffer = _io.BytesIO(cached_index_data)
            faiss_index = faiss.deserialize_index(faiss.read_index(index_buffer))
            registry = get_chunk_registry()
            st.info(
                f"📂 Loaded FAISS index from Redis cache with {faiss_index.ntotal} vectors"
            )
        except Exception as e:
            print(f"[Redis] Error loading cached index: {e}, falling back to disk")
            from src.core.faiss_index import load_or_rebuild_index

            faiss_index, registry, index_recovered = load_or_rebuild_index(_INDEX_PATH)

            if index_recovered:
                if faiss_index.ntotal:
                    st.warning(
                        "FAISS index was missing, corrupted, or inconsistent and was "
                        f"automatically rebuilt from {faiss_index.ntotal} stored vectors."
                    )
                else:
                    st.info(
                        "No stored embeddings were found. An empty FAISS index was "
                        "initialized safely."
                    )
            else:
                st.info(
                    f"Loaded and validated the existing FAISS index with "
                    f"{faiss_index.ntotal} vectors."
                )
    else:
        if os.path.exists(_INDEX_PATH):
            faiss_index = load_index(_INDEX_PATH)
            registry = get_chunk_registry()
        else:
            faiss_index = None
            registry = []

    if SessionKeys.ANALYSIS_RESULTS not in st.session_state:
        st.session_state[SessionKeys.ANALYSIS_RESULTS] = None
        cached_results = get_analysis_results(f"{SESSION_ID}:current")
        if cached_results is not None:
            st.session_state[SessionKeys.ANALYSIS_RESULTS] = cached_results

    if SessionKeys.ANALYSIS_FILE_SIGNATURE not in st.session_state:
        st.session_state[SessionKeys.ANALYSIS_FILE_SIGNATURE] = None

        cached_signature = get_session_state(SESSION_ID, "analysis_file_signature")
        if cached_signature is not None:
            st.session_state[SessionKeys.ANALYSIS_FILE_SIGNATURE] = cached_signature
            faiss_index = (
                load_index(_INDEX_PATH) if os.path.exists(_INDEX_PATH) else None
            )
            registry = get_chunk_registry()
    else:
        faiss_index = load_index(_INDEX_PATH) if os.path.exists(_INDEX_PATH) else None

    uploaded_files = st.file_uploader(
        get_text("upload_title", lang=lang_code),
        type=["pdf", "docx", "txt", "md", "markdown", "mdown", "zip", "csv"],
        accept_multiple_files=True,
        key="file_uploader",
    )

    MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10MB limit
    file_bytes_dict = {}

    if bulk_download_drive_folder is not None:
        with st.expander("📁 Import from Google Drive", expanded=False):
            drive_folder_input = st.text_input(
                "Google Drive folder URL or ID",
                key="drive_folder_input",
                placeholder="https://drive.google.com/drive/folders/…",
            )
            drive_api_key = st.text_input(
                "Google Drive API key",
                key="drive_api_key",
                type="password",
                help="Optional if GOOGLE_DRIVE_API_KEY is set in the environment.",
            )
            if st.button("Import from Drive", key="drive_import_btn"):
                if not drive_folder_input:
                    st.error("Please enter a Google Drive folder URL or ID.")
                else:
                    drive_progress_bar = st.progress(
                        0, text="Connecting to Google Drive…"
                    )

                    def _update_drive_progress(bytes_downloaded, total_bytes):
                        fraction = (
                            min(bytes_downloaded / total_bytes, 1.0)
                            if total_bytes
                            else 0
                        )
                        drive_progress_bar.progress(
                            fraction,
                            text=(
                                f"Downloading from Drive… "
                                f"{bytes_downloaded / 1024:.0f} KB"
                                + (
                                    f" / {total_bytes / 1024:.0f} KB"
                                    if total_bytes
                                    else ""
                                )
                            ),
                        )

                    try:
                        drive_files, drive_names = bulk_download_drive_folder(
                            drive_folder_input,
                            api_key=drive_api_key or None,
                            progress_callback=_update_drive_progress,
                        )
                        st.session_state.setdefault("drive_imported_files", {})
                        st.session_state["drive_imported_files"].update(drive_files)
                        drive_progress_bar.progress(
                            1.0, text=f"Imported {len(drive_names)} file(s)."
                        )
                        st.success(
                            f"Imported {len(drive_names)} file(s) from Google Drive: "
                            f"{', '.join(drive_names)}"
                        )
                    except Exception as exc:
                        drive_progress_bar.empty()
                        st.error(f"⚠️ Google Drive import failed: {exc}")

    if uploaded_files:
        for uploaded_file in uploaded_files:
            original_name = uploaded_file.name
            try:
                validate_document_extension(
                    original_name,
                    allowed_extensions={
                        ".csv",
                        ".docx",
                        ".md",
                        ".markdown",
                        ".mdown",
                        ".pdf",
                        ".txt",
                        ".zip",
                    },
                )
            except InvalidFileExtensionError as exc:
                st.error(
                    f"⚠️ File **'{sanitize_filename(original_name)}'** was rejected: {exc}"
                )
                continue

            safe_name = unique_filename(original_name, file_bytes_dict)

            if uploaded_file.size > MAX_FILE_SIZE_BYTES:
                st.error(
                    f"⚠️ File **'{safe_name}'** exceeds maximum size limit of 10MB."
                )
                continue

            file_hash = compute_file_hash_stream(uploaded_file)
            uploaded_file.seek(0)
            existing_doc = get_document_by_hash(file_hash)

            if existing_doc:
                st.warning(f"⚠️ File **'{original_name}'** is identical to **'{existing_doc}'** already in the database.")
                action = st.radio(
                    f"Action for duplicate file '{original_name}':",
                    ["Skip", "Reprocess"],
                    key=f"dup_{file_hash}_{original_name}",
                    horizontal=True
                )
                if action == "Skip":
                    continue

            file_bytes = uploaded_file.read()
            file_bytes_dict[safe_name] = strip_exif_metadata(
                file_bytes, safe_name
            )

    for drive_name, drive_bytes in st.session_state.get(
        "drive_imported_files", {}
    ).items():
        safe_drive_name = unique_filename(drive_name, file_bytes_dict)
        file_bytes_dict[safe_drive_name] = drive_bytes

    has_enough_files = len(file_bytes_dict) >= 2

    @st.cache_data(show_spinner=False)
    def run_extraction_pipeline(
        raw_texts_items: tuple,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        raw_texts_dict = dict(raw_texts_items)
        chunked_docs = chunk_documents(
            raw_texts_dict, chunk_size=chunk_size, chunk_overlap=chunk_overlap
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
        faiss_index, registry = build_index(embeddings, chunked_docs)
        ai_probabilities = detect_documents_ai_probability(chunked_docs)

        return (
            chunked_docs,
            embeddings,
            sim_df,
            chunk_sim_df,
            faiss_index,
            registry,
            ai_probabilities,
        )

    if has_enough_files:
        st.session_state[SessionKeys.SCANNING] = True
        total_bytes = sum(len(data) for data in file_bytes_dict.values())
        file_count = len(file_bytes_dict)

        progress_bar = st.progress(0, text="Preparing files…")
        raw_texts = {}
        for i, (name, data) in enumerate(file_bytes_dict.items()):
            raw_texts[name] = extract_text(
                _io.BytesIO(data), name, ocr_language=ocr_language, ocr_dpi=ocr_dpi
            )
            fraction = (i + 1) / file_count
            remaining_bytes = total_bytes * (file_count - i - 1) // max(1, file_count)
            remaining_est = estimate_processing_seconds(remaining_bytes)
            eta = (
                format_processing_duration(remaining_est)
                if remaining_est
                else "a moment"
            )
            progress_bar.progress(
                fraction,
                text=f"Processing file {i + 1} of {file_count} (ETA: {eta})",
            )

        raw_texts_tuple = tuple(sorted(raw_texts.items()))
        (
            chunked_docs,
            embeddings,
            sim_df,
            chunk_sim_df,
            faiss_index,
            registry,
            ai_probabilities,
        ) = run_extraction_pipeline(
            raw_texts_tuple,
            chunk_size,
            chunk_overlap,
        )
        st.session_state[SessionKeys.SCANNING] = False
        active_sim_df = chunk_sim_df if use_chunk_matrix else sim_df
        flags = flag_plagiarism(active_sim_df, threshold=threshold)

        init_incident_db()
        incidents = sync_flagged_incidents(flags)
    else:
        flags = []
        active_sim_df = None
        raw_texts = {}
        ai_probabilities = {}

st.subheader(get_text("analysis_summary", lang=lang_code))
doc_names = list(raw_texts.keys())
n_docs = len(doc_names)
total_pairs = n_docs * (n_docs - 1) // 2 if n_docs > 1 else 0
n_flagged = len(flags)
total_doc_count = max(n_docs, get_total_document_count())

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Documents", total_doc_count)
col2.metric("Pairs Evaluated", total_pairs)
col3.metric("Flagged Pairs", n_flagged)
col4.metric("FAISS Vectors", faiss_index.ntotal if faiss_index is not None else 0)
col5.metric("🎯 Threshold", f"{threshold:.0%}")
st.divider()

(
    tab_warnings,
    tab_faiss,
    tab_matrix,
    tab_heatmap,
    tab_drill,
    tab_analytics,
    tab_users,
    tab_settings,
    tab_history,
    tab_audit,
) = st.tabs(
    [
        get_text("tab_warnings", lang=lang_code),
        get_text("tab_faiss", lang=lang_code),
        get_text("tab_matrix", lang=lang_code),
        get_text("tab_heatmap", lang=lang_code),
        get_text("tab_drill", lang=lang_code),
        get_text("tab_analytics", lang=lang_code),
        get_text("tab_users", lang=lang_code),
        get_text("tab_settings", lang=lang_code),
        get_text("tab_audit_logs", lang=lang_code),
        "📊 History",
    ],
    key="main_tabs",
)

# Record scan summary for historical tracking
if flags and len(file_bytes_dict) >= 2:
    from src.db.corpus_db import record_scan_summary
    
    all_sims = [f["similarity"] for f in flags]
    avg_sim = sum(all_sims) / len(all_sims) if all_sims else 0.0
    max_sim = max(all_sims) if all_sims else 0.0
    
    record_scan_summary(
        document_count=len(file_bytes_dict),
        avg_similarity=avg_sim,
        max_similarity=max_sim,
        flagged_count=len(flags),
        threshold_used=threshold,
    )


# ══ TAB 1: WARNINGS ═══════════════════════════════════════════════════════
with tab_warnings:
    update_page_title("Warnings")
    st.subheader(get_text("tab_warnings", lang=lang_code))

    auto_refresh_enabled = st.toggle(
        "Auto-refresh live feed (30s)",
        value=False,
        key=SessionKeys.INCIDENT_STREAM_AUTO_REFRESH,
        help=(
            "When enabled, the incident feed re-runs every 30 seconds "
            "to surface newly flagged submissions automatically."
        ),
    )

    if auto_refresh_enabled and st_autorefresh is not None:
        st_autorefresh(
            interval=30 * 1000,
            key="incident_stream_autorefresh",
        )

    st.session_state[SessionKeys.INCIDENT_STREAM_AUTO_REFRESH] = auto_refresh_enabled

    if auto_refresh_enabled:
        if st_autorefresh is None:
            st.warning(
                "Auto-refresh is enabled, but the `streamlit-autorefresh` "
                "package is not installed. Install it via "
                "`pip install streamlit-autorefresh`."
            )
        else:
            st.caption("🔴 Live — refreshing every 30 seconds.")
    else:
        st.caption("⚪ Live feed paused — toggle on to auto-refresh.")

    st.divider()

    if SessionKeys.WARNINGS_EXPAND_ALL not in st.session_state:
        st.session_state[SessionKeys.WARNINGS_EXPAND_ALL] = False

    st.markdown("### 📅 Incident Date Filter")
    date_preset = st.radio(
        "Select Date Range",
        options=["Today", "Last 7 Days", "Last 30 Days", "All Time"],
        horizontal=True,
        key="incident_date_preset",
        help="Quickly filter the incident table by common date ranges.",
    )

    start_date, end_date = get_date_range_preset(date_preset)
    st.caption(
        f"Filtering incidents from **{start_date.strftime('%Y-%m-%d')}** to "
        f"**{end_date.strftime('%Y-%m-%d')}**"
    )

    if not flags:
        st.info("No plagiarism incidents detected above configured threshold.")
    elif render_warning_controls is not None:
        render_warning_controls(
            flags, threshold=threshold, ai_probabilities=ai_probabilities
        )

        button_label = (
            "📂 Expand All"
            if not st.session_state[SessionKeys.WARNINGS_EXPAND_ALL]
            else "📁 Collapse All"
        )

        if st.button(button_label, key="toggle_warning_accordions"):
            st.session_state[SessionKeys.WARNINGS_EXPAND_ALL] = not st.session_state[
                SessionKeys.WARNINGS_EXPAND_ALL
            ]
            st.rerun()

        render_warning_controls(
            flags,
            threshold=threshold,
            ai_probabilities=ai_probabilities,
            expanded=st.session_state[SessionKeys.WARNINGS_EXPAND_ALL],
        )

# ══ TAB 2: FAISS ══════════════════════════════════════════════════════════
with tab_faiss:
    update_page_title("FAISS")
    st.subheader("⚡ FAISS Vector Search")
    if faiss_index is not None:
        st.info(f"Index total: {faiss_index.ntotal} vectors.")
        faiss_query = st.text_input("Query FAISS Index:", key="faiss_query_input")
        if st.button("Run Search") and faiss_query.strip():
            q_vec = embed_chunks([faiss_query.strip()])[0]
            results = search_similar_chunks(
                q_vec, faiss_index, registry, top_k=faiss_top_k, threshold=threshold
            )
            from app.components.faiss_results import render_faiss_results_ui

            render_faiss_results_ui(results, faiss_query.strip(), document_pdf_bytes=globals().get("file_bytes_dict"))


# ══ TAB 3: MATRIX ═════════════════════════════════════════════════════════
with tab_matrix:
    update_page_title("Matrix")
    st.subheader("📋 Similarity Matrix")
    if active_sim_df is not None:
        st.dataframe(active_sim_df.style.format("{:.4f}"), use_container_width=True)

# ══ TAB 4: HEATMAP ════════════════════════════════════════════════════════
with tab_heatmap:
    update_page_title("Heatmap")
    st.subheader("🗺️ Heatmap & Network")
    heatmap_fig = None
    if active_sim_df is not None:
        heatmap_fig = ui_exception_handler("Similarity Heatmap")(
            plot_similarity_heatmap
        )(active_sim_df, threshold=threshold, theme_colors=get_chart_colors())

    if heatmap_fig is not None:
        # plot_similarity_heatmap() returns a Matplotlib Figure, so it is
        # rendered with st.pyplot(), not st.plotly_chart(). Passing
        # use_container_width=True keeps it in sync with the container
        # width (sidebar collapse/expand, mobile/tablet/desktop layouts).
        st.pyplot(heatmap_fig, use_container_width=True)

    doc_select_options = (
        ["None"] + list(active_sim_df.columns)
        if active_sim_df is not None
        else ["None"]
    )
    selected_highlight_doc = st.selectbox(
        "Highlight Document Node",
        options=doc_select_options,
        index=0,
        key="highlight_doc_node_selector",
    )
    highlighted_doc = (
        selected_highlight_doc if selected_highlight_doc != "None" else None
    )

    network_fig = None
    if active_sim_df is not None:
        network_fig = ui_exception_handler("Plagiarism Network")(
            plot_similarity_network
        )(
            similarity_df=active_sim_df,
            threshold=threshold,
            highlighted_doc=highlighted_doc,
            title="Interactive Document Plagiarism Network",
        )

    if network_fig is not None:
        # plot_similarity_network() returns a Plotly go.Figure.
        st.plotly_chart(network_fig, use_container_width=True)

    # ── Plagiarism Cluster Detection Summary (Issue #1675) ───────────────────
    if active_sim_df is not None and len(doc_names) >= 2:
        from src.core.similarity import detect_plagiarism_clusters
        
        cluster_data = detect_plagiarism_clusters(active_sim_df, threshold=threshold)
        suspicious_groups = cluster_data["suspicious_groups"]
        
        if suspicious_groups:
            with st.expander(
                f"🚨 Suspicious Collusion Rings Detected ({len(suspicious_groups)})",
                expanded=True,
            ):
                st.warning(
                    f"Found {len(suspicious_groups)} group(s) of 3+ highly similar documents. "
                    "These may indicate collusion or shared source material."
                )
                
                for group in suspicious_groups:
                    st.markdown(f"**Cluster #{group['cluster_id']}** ({group['size']} documents):")
                    for doc in group["documents"]:
                        st.markdown(f"- 📄 `{doc}`")
                    st.divider()


# ══ TAB 5: PAIR DRILL-DOWN ════════════════════════════════════════════════
with tab_drill:
    update_page_title("Drill Down")
    st.subheader("🔬 Pair Drill-Down")

    # ── Issue #1383: Cosine vs Lexical side-by-side comparison table ──
    render_cosine_vs_lexical_comparison_table(
        active_sim_df,
        raw_texts,
        semantic_threshold=SEMANTIC_HIGH_THRESHOLD,
        lexical_threshold=LEXICAL_LOW_THRESHOLD,
    )

    st.markdown("---")

    if active_sim_df is not None and len(doc_names) >= 2:
        c1, c2 = st.columns(2)
        with c1:
            da = st.selectbox("Document A", doc_names, key="da")
        with c2:
            db = st.selectbox("Document B", [d for d in doc_names if d != da], key="db")
        sim_val = float(active_sim_df.loc[da, db])
        st.write(f"Overall Similarity: `{sim_val:.1%}`")

        pair_flags = [
            f
            for f in flags
            if (f["doc_a"] == da and f["doc_b"] == db)
            or (f["doc_a"] == db and f["doc_b"] == da)
        ]

        if pair_flags:
            st.markdown("### 📝 Flagged Snippets")
            for rank, flag in enumerate(pair_flags, 1):
                ca = str(flag.get("snippet_a", ""))
                cb = str(flag.get("snippet_b", ""))

                if flag["doc_a"] == db:
                    ca, cb = cb, ca

                highlighted_ca, highlighted_cb = highlight_overlap(ca, cb)

                with st.expander(
                    f"Incident #{rank} - Similarity: {flag.get('similarity', 0.0):.1%}",
                    expanded=(rank == 1),
                ):
                    c_a, c_b = st.columns(2)
                    with c_a:
                        st.markdown(f"**{da}**")
                        st.markdown(highlighted_ca, unsafe_allow_html=True)
                        if render_copy_button:
                            render_copy_button(
                                text_to_copy=ca,
                                button_id=f"copy_ca_{rank}",
                                copy_label="📋 Copy Snippet",
                            )
                    with c_b:
                        st.markdown(f"**{db}**")
                        st.markdown(highlighted_cb, unsafe_allow_html=True)
                        if render_copy_button:
                            render_copy_button(
                                text_to_copy=cb,
                                button_id=f"copy_cb_{rank}",
                                copy_label="📋 Copy Snippet",
                            )

# ══ TAB 6: ANALYTICS ══════════════════════════════════════════════════════
with tab_analytics:
    update_page_title("Analytics")
    st.subheader("📊 Analytics Dashboard")
    st.markdown("### ⏱️ Pipeline Processing Time Breakdown")
    stage_timings = st.session_state.get("last_stage_timings") or st.session_state.get("stage_timings")
    if plot_processing_time_breakdown:
        active_theme_colors = get_chart_colors() if callable(get_chart_colors) else None
        fig_time = plot_processing_time_breakdown(
            stage_timings=stage_timings,
            theme_colors=active_theme_colors,
        )
        st.plotly_chart(fig_time, use_container_width=True)
    else:
        st.info("Analytics metrics summary loaded.")

# ══ TAB 7: USERS ══════════════════════════════════════════════════════════
with tab_users:
    update_page_title("Users")
    st.subheader("👥 User Management")
    users = get_all_users()
    for u in users:
        st.write(f"User: **{u['username']}** | Role: `{u['role']}`")

# ══ TAB 8: SETTINGS ═══════════════════════════════════════════════════════
with tab_settings:
    update_page_title("Settings")
    st.subheader("⚙️ System Configuration")

    from app.components.storage_quota import render_storage_quota_progress
    render_storage_quota_progress()

    st.markdown("### 📊 Visualization")
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
                import subprocess
                import sys

                seed_script = os.path.join(ROOT_DIR, "scripts", "generate_seed_data.py")
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
        threshold = st.slider(
            get_text("threshold", lang=lang_code),
            min_value=0.0,
            max_value=1.0,
            value=DEFAULT_THRESHOLDS.plagiarism,
            step=0.01,
            help=(
                "Combined Hybrid score threshold for flagging pair plagiarism. "
                "Calculated from Lexical (exact phrase overlap) and Semantic (meaning alignment) scores. "
                "Recommended Default: 0.59 (59%)."
            ),
            key=SessionKeys.THRESHOLD_SLIDER,
            on_change=save_preferences_callback,
        )

        lexical_threshold = st.slider(
            "Lexical Sensitivity Threshold",
            0.0,
            1.0,
            value=0.50,
            step=0.01,
            help=(
                "Direct word-for-word and N-gram match threshold. "
                "Higher values require near-identical text phrasing to trigger alerts. "
                "Recommended Default: 0.50 (50%)."
            ),
            key="settings_lexical_slider",
        )

        semantic_threshold = st.slider(
            "Semantic Sensitivity Threshold",
            0.0,
            1.0,
            value=0.65,
            step=0.01,
            help=(
                "Transformer embedding vector similarity threshold measuring conceptual alignment and paraphrasing. "
                "Higher values require strong contextual similarity even if words differ. "
                "Recommended Default: 0.65 (65%)."
            ),
            key="settings_semantic_slider",
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
                key=SessionKeys.OCR_LANGUAGE_SELECTOR,
            )
            ocr_language = ocr_language_labels[selected_ocr_language_name]

            ocr_dpi = st.slider(
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
            "🔍 Ping Redis", key="ping_redis_button", use_container_width=True
        ):
            from src.utils.redis_cache import get_cache

            connected, latency = get_cache().ping()
            if connected:
                st.success(f"✅ Connected ({latency} ms ping)")
            else:
                st.error("🚨 Disconnected")

        st.markdown("### 🗄️ Database Schema Status")
        if st.button("Check Database Schema", key="check_db_schema_btn", use_container_width=True):
            try:
                import sqlite3
                from src.core.app_config import CORPUS_DB_PATH, AUTH_DB_PATH
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

                st.session_state["db_schema_status_msg"] = f"Corpus Schema: v{corpus_ver} | Auth Schema: v{auth_ver}"
                st.toast("✅ Database schema checked successfully!")
            except Exception as e:
                st.error(f"❌ Failed to check schema versions: {e}")

        if "db_schema_status_msg" in st.session_state:
            st.info(st.session_state["db_schema_status_msg"])
            
# ══ TAB 9: SECURITY AUDIT LOGS ═════════════════════════════════════════════
with tab_audit:
    update_page_title("Security Audit Logs")
    st.subheader(get_text("tab_audit_logs", lang=lang_code))

    if user_role != "admin":
        st.error(
            "🔒 Access Denied: Administrator privileges required to view security audit logs."
        )
    else:
        st.markdown("### 📜 System Security Audit Trail")

        # Filters section
        filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)

        with filter_col1:
            date_range = st.date_input(
                "📅 Date Range Filter",
                value=(),
                key="audit_date_range_picker",
                help="Filter audit log records by date range.",
            )

        start_date_str = None
        end_date_str = None
        if isinstance(date_range, (list, tuple)) and len(date_range) > 0:
            if len(date_range) == 1:
                start_date_str = date_range[0].strftime("%Y-%m-%d") + "T00:00:00Z"
                end_date_str = date_range[0].strftime("%Y-%m-%d") + "T23:59:59Z"
            elif len(date_range) == 2:
                start_date_str = date_range[0].strftime("%Y-%m-%d") + "T00:00:00Z"
                end_date_str = date_range[1].strftime("%Y-%m-%d") + "T23:59:59Z"

        with filter_col2:
            distinct_events = get_distinct_audit_event_types()
            event_type_options = ["All Event Types"] + distinct_events
            selected_event_type = st.selectbox(
                "🏷️ Event Type",
                options=event_type_options,
                key="audit_event_type_filter",
            )
            event_type_filter = (
                None
                if selected_event_type == "All Event Types"
                else selected_event_type
            )

        with filter_col3:
            username_filter_input = st.text_input(
                "👤 Filter by Username",
                value="",
                placeholder="Enter username...",
                key="audit_username_filter",
            ).strip()
            username_filter = username_filter_input if username_filter_input else None

        with filter_col4:
            per_page = st.selectbox(
                "📄 Rows Per Page",
                options=[10, 25, 50, 100],
                index=1,  # Default 25
                key="audit_per_page_select",
            )

        # Count total matching records
        total_records = get_security_audit_log_count(
            username=username_filter,
            event_type=event_type_filter,
            start_date=start_date_str,
            end_date=end_date_str,
        )

        total_pages = max(1, (total_records + per_page - 1) // per_page)

        current_page = st.session_state.get(SessionKeys.AUDIT_LOG_PAGE, 1)
        if current_page > total_pages:
            current_page = total_pages
            st.session_state[SessionKeys.AUDIT_LOG_PAGE] = current_page
        if current_page < 1:
            current_page = 1
            st.session_state[SessionKeys.AUDIT_LOG_PAGE] = current_page

        offset = (current_page - 1) * per_page

        # Fetch records for current page
        logs = get_security_audit_logs(
            username=username_filter,
            event_type=event_type_filter,
            start_date=start_date_str,
            end_date=end_date_str,
            limit=per_page,
            offset=offset,
        )

        # Summary Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("📋 Total Log Entries", total_records)
        m2.metric("🏷️ Active Filter", selected_event_type)
        m3.metric("📑 Page", f"{current_page} / {total_pages}")

        st.divider()

        # Display Data Table
        if logs:
            df = pd.DataFrame(logs)
            display_df = df[
                ["id", "timestamp", "event_type", "username", "details"]
            ].rename(
                columns={
                    "id": "ID",
                    "timestamp": "Timestamp (UTC)",
                    "event_type": "Event Type",
                    "username": "Username",
                    "details": "Details / Payload",
                }
            )

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "ID": st.column_config.NumberColumn("ID", width="small"),
                    "Timestamp (UTC)": st.column_config.TextColumn(
                        "Timestamp (UTC)", width="medium"
                    ),
                    "Event Type": st.column_config.TextColumn(
                        "Event Type", width="medium"
                    ),
                    "Username": st.column_config.TextColumn("Username", width="medium"),
                    "Details / Payload": st.column_config.TextColumn(
                        "Details / Payload", width="large"
                    ),
                },
            )

            # Pagination Controls
            nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([1, 2, 2, 1])
            with nav_col1:
                if st.button(
                    "← Previous",
                    disabled=(current_page <= 1),
                    key="audit_prev_page",
                ):
                    st.session_state[SessionKeys.AUDIT_LOG_PAGE] = current_page - 1
                    st.rerun()

            with nav_col2:
                end_range = min(offset + per_page, total_records)
                start_range = offset + 1 if total_records > 0 else 0
                st.caption(
                    f"Showing {start_range} - {end_range} of {total_records} logs"
                )

            with nav_col3:
                page_select = st.number_input(
                    "Go to Page",
                    min_value=1,
                    max_value=total_pages,
                    value=current_page,
                    step=1,
                    key="audit_page_num_input",
                )
                if page_select != current_page:
                    st.session_state[SessionKeys.AUDIT_LOG_PAGE] = page_select
                    st.rerun()

            with nav_col4:
                if st.button(
                    "Next →",
                    disabled=(current_page >= total_pages),
                    key="audit_next_page",
                ):
                    st.session_state[SessionKeys.AUDIT_LOG_PAGE] = current_page + 1
                    st.rerun()

            st.divider()

            # CSV Export Functionality
            export_all_logs = get_security_audit_logs(
                username=username_filter,
                event_type=event_type_filter,
                start_date=start_date_str,
                end_date=end_date_str,
                limit=10000,
                offset=0,
            )
            export_df = pd.DataFrame(export_all_logs)
            csv_bytes = export_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="⬇️ Download Audit Logs (CSV)",
                data=csv_bytes,
                file_name=f"security_audit_logs_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="download_audit_logs_csv",
                use_container_width=True,
                type="primary",
            )
        else:
            st.info(
                "ℹ️ No security audit log records found matching the specified filters."
            )

# ══ TAB 10: History ══════════════════════════════════════════════════════════
with tab_history:
    update_page_title("History")
    st.subheader("📊 Document Similarity History Dashboard")
    st.caption("Monitor plagiarism patterns and similarity trends across previous scan sessions.")
    
    from src.db.corpus_db import get_scan_history
    from src.visualization.history_charts import plot_similarity_trend_line, plot_flagged_documents_bar
    from datetime import datetime, timedelta
    
    # Date range filter
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=30),
            key="history_start_date",
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now(),
            key="history_end_date",
        )
        
    history_data = get_scan_history(
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        limit=100,
    )
    
    if not history_data:
        st.info("No scan history found for the selected date range. Run a scan to populate this dashboard.")
    else:
        # Similarity Trend Line Chart
        trend_fig = plot_similarity_trend_line(history_data, theme_colors=get_chart_colors())
        st.plotly_chart(trend_fig, use_container_width=True)
        
        st.divider()
        
        # Flagged Documents Bar Chart
        bar_fig = plot_flagged_documents_bar(history_data, theme_colors=get_chart_colors())
        st.plotly_chart(bar_fig, use_container_width=True)
        
        st.divider()
        
        # Raw Data Table
        st.markdown("### 📋 Raw Scan History Data")
        df_history = pd.DataFrame(history_data)
        df_history["timestamp"] = pd.to_datetime(df_history["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        st.dataframe(
            df_history.style.format({
                "avg_similarity": "{:.2%}",
                "max_similarity": "{:.2%}",
                "threshold_used": "{:.2%}",
            }),
            use_container_width=True,
            hide_index=True,
        )

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
from src.utils.version_check import APP_VERSION, check_for_update_sync

if "_update_check_tag" not in st.session_state:
    st.session_state["_update_check_tag"] = check_for_update_sync(APP_VERSION)

_latest_tag: str | None = st.session_state["_update_check_tag"]

_footer_col1, _footer_col2 = st.columns([3, 1])
with _footer_col1:
    st.caption(
        f"🎓 Semantic Plagiarism Detection System · v{APP_VERSION} · Streamlit · "
        "🐛 Report Bug / Feedback"
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
        