"""
Streamlit App State Manager.

Handles session state initialization, session timeout enforcement,
user activity tracking, preference callbacks, background server/backup daemons,
and UI exception wrapping.
"""

import functools
import logging
import os
import threading
import time
import traceback
import uuid
import os
import multiprocessing
from datetime import datetime

import streamlit as st

from app.session_keys import SessionKeys
from src.core.config import PLAGIARISM_THRESHOLD
from src.db.auth import update_user_preferences
from src.utils.redis_cache import (
    cache_session_state,
    clear_session,
    get_cache,
    get_session_state,
)

logger = logging.getLogger(__name__)

TIMEOUT_LIMIT = 15 * 60  # 15 minutes in seconds

# Module-level lock to ensure thread-safe daemon initialization
_daemon_init_lock = threading.Lock()


def ui_exception_handler(component_name: str):
    """Decorator that catches exceptions in a UI component and shows a
    friendly error message instead of a raw Streamlit traceback."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except st.runtime.scriptrunner.StopException:
                raise
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


def update_global_activity():
    """Update the global last_activity timestamp."""
    try:
        cache = get_cache()
        cache.set("spd:v1:global:last_activity", time.time())
    except Exception as e:
        logger.error(f"Failed to update global activity: {e}")


def get_active_sessions_count() -> int:
    """Return the number of active Streamlit sessions, or -1 on failure."""
    try:
        cache = get_cache()
        if cache is None:
            return -1

        now = time.time()
        active_count = 0
        keys = []
        scan_failed = False

        if cache.is_available():
            try:
                raw_keys = list(
                    cache._client.scan_iter(match="spd:v1:session:*:last_interaction")
                )
                keys = [
                    k.decode("utf-8") if isinstance(k, bytes) else k for k in raw_keys
                ]
            except Exception as e:
                logger.error(f"Failed to scan Redis session keys: {e}")
                scan_failed = True

        try:
            fallback_dict = getattr(cache, "fallback_cache", {})
            if fallback_dict is not None:
                fallback_keys = [
                    k
                    for k in list(fallback_dict.keys())
                    if k.startswith("spd:v1:session:")
                    and k.endswith(":last_interaction")
                ]
                for k in fallback_keys:
                    if k not in keys:
                        keys.append(k)
        except Exception as e:
            logger.error(f"Failed to scan fallback cache session keys: {e}")
            if not cache.is_available() or scan_failed:
                return -1

        if scan_failed and not keys:
            return -1

        for key in keys:
            try:
                parts = key.split(":")
                if len(parts) >= 4:
                    session_id = parts[3]
                    last_interaction = get_session_state(
                        session_id,
                        SessionKeys.LAST_INTERACTION,
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
        return -1


def _start_api_server():
    import uvicorn

    from starlette.middleware.base import BaseHTTPMiddleware

    from src.api.app import app as fastapi_app

    class ActivityMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            if request.url.path not in ("/health", "/healthz"):
                update_global_activity()
            return await call_next(request)

    fastapi_app.add_middleware(ActivityMiddleware)

    from src.core.app_config import API_PORT

    uvicorn.run(
        fastapi_app,
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=API_PORT,
        log_level="warning",
    )

def init_api_server_daemon():
    """Ensure background REST API server is started once in a thread-safe manner."""
    import src.core.app_config as app_config

    if not getattr(app_config, "_api_server_started", False):
        app_config._api_server_started = True

        app_config.api_server_process = multiprocessing.Process(
            target=_start_api_server,
            daemon=True,
        )
        app_config.api_server_process.start()


def _run_backup_daemon():
    """Background loop to create backups after inactivity."""
    last_backup_time = 0.0
    daemon_start_time = time.time()

    try:
        cache = get_cache()
        cached = cache.get("spd:v1:global:last_backup_time")
        if cached is not None:
            last_backup_time = float(cached)
    except Exception:
        pass

    logger.info("Database backup daemon started.")

    try:
        cache = get_cache()
        if cache.get("spd:v1:global:last_activity") is None:
            cache.set("spd:v1:global:last_activity", time.time())
    except Exception:
        pass

    while True:
        time.sleep(30)

        try:
            from src.core.app_config import get_backup_idle_timeout

            cache = get_cache()
            timeout = get_backup_idle_timeout()
            now = time.time()
            
            # Establish a safe startup state: wait for at least one timeout interval
            # after daemon startup before allowing ANY backups to trigger.
            # We track this explicit condition rather than using an arbitrary sleep/continue.
            is_startup_phase = (now - daemon_start_time) < timeout

            last_activity = cache.get("spd:v1:global:last_activity")
            if last_activity is None:
                last_activity = now
                cache.set("spd:v1:global:last_activity", last_activity)
            else:
                try:
                    last_activity = float(last_activity)
                except (ValueError, TypeError):
                    last_activity = now
                    cache.set("spd:v1:global:last_activity", last_activity)

            idle = now - last_activity

            active_sessions = get_active_sessions_count()
            if active_sessions < 0:
                logger.warning(
                    "Skipping automated database backup due to active sessions count failure (%d).",
                    active_sessions,
                )
                continue

            if (
                active_sessions == 0
                and idle >= timeout
                and last_activity > last_backup_time
                and not is_startup_phase
            ):
                from src.core.app_config import get_backup_dir
                from src.db.database_backup import (
                    cleanup_old_backups,
                    create_corpus_database_snapshot,
                )

                snapshot = create_corpus_database_snapshot()

                backup_dir = get_backup_dir()
                backup_dir.mkdir(parents=True, exist_ok=True)

                filename = (
                    backup_dir / f"corpus_backup_{datetime.now():%Y%m%d_%H%M%S}.db"
                )

                filename.write_bytes(snapshot)

                logger.info(f"Backup created: {filename}")
                cleanup_old_backups(backup_dir, max_backups=10, max_age_days=30)

                last_backup_time = now
                cache.set(
                    "spd:v1:global:last_backup_time",
                    last_backup_time,
                )

        except Exception as e:
            logger.exception(f"Backup daemon error: {e}")


def init_backup_daemon():
    """Ensure database backup daemon thread is running in a thread-safe manner."""
    import src.core.app_config as app_config

    with _daemon_init_lock:
        if not getattr(app_config, "_backup_daemon_started", False):
            app_config._backup_daemon_started = True
            threading.Thread(
                target=_run_backup_daemon,
                daemon=True,
            ).start()


def init_session_state():
    """Initialize session state keys and global background services."""
    import app.session_manager as sm

    st.session_state[SessionKeys.SESSION_ID] = sm.initialize_and_verify_session()

    if SessionKeys.AUTHENTICATED not in st.session_state:
        st.session_state[SessionKeys.AUTHENTICATED] = False
    if SessionKeys.USERNAME not in st.session_state:
        st.session_state[SessionKeys.USERNAME] = None
    if SessionKeys.PDF_PASSWORDS not in st.session_state:
        st.session_state[SessionKeys.PDF_PASSWORDS] = {}
    if SessionKeys.LANG not in st.session_state:
        st.session_state[SessionKeys.LANG] = "en"
    if SessionKeys.SESSION_START_TIME not in st.session_state:
        st.session_state[SessionKeys.SESSION_START_TIME] = time.time()

    if SessionKeys.MODEL_LOAD_TIME not in st.session_state:
        from src.core.embedding_model import EmbeddingModelManager

        with st.spinner("Initializing Vector Embedding Model..."):
            _start_time = time.perf_counter()
            EmbeddingModelManager.get_instance().get_model()
            st.session_state[SessionKeys.MODEL_LOAD_TIME] = (
                time.perf_counter() - _start_time
            )

    update_global_activity()
    init_api_server_daemon()
    init_backup_daemon()

    return st.session_state[SessionKeys.SESSION_ID]


def check_session_timeout(session_id: str):
    """Enforce 15-minute inactivity session timeout limit."""
    cached_last_interaction = get_session_state(
        session_id, SessionKeys.LAST_INTERACTION
    )
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
            clear_session(session_id)
            from src.errors import UI_SESSION_EXPIRED

            st.warning(UI_SESSION_EXPIRED)
            st.stop()
        else:
            st.session_state[SessionKeys.LAST_INTERACTION] = time.time()
            cache_session_state(session_id, SessionKeys.LAST_INTERACTION, time.time())

    return last_interaction


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


def reset_analysis_data() -> None:
    """Clear document analysis and scan results from session state while preserving authentication and theme preferences."""
    preserved_keys = {
        SessionKeys.AUTHENTICATED,
        SessionKeys.USERNAME,
        SessionKeys.ROLE,
        SessionKeys.SESSION_ID,
        SessionKeys.LANG,
        SessionKeys.SESSION_START_TIME,
        SessionKeys.MODEL_LOAD_TIME,
        SessionKeys.LAST_INTERACTION,
        SessionKeys.ACCENT_COLOR,
        SessionKeys.COMPACT_VIEW,
        SessionKeys.FORCE_DARK_CHARTS,
        SessionKeys.PDF_PASSWORDS,
        "authenticated",
        "username",
        "role",
        "session_id",
        "lang",
        "session_start_time",
        "model_load_time",
        "last_interaction",
        "accent_color",
        "compact_view",
        "force_dark_charts",
        "pdf_passwords",
        "theme_selector",
        "theme",
        "active_session_token",
        "raw_session_uuid",
    }

    target_keys = {
        SessionKeys.ANALYSIS_RESULTS,
        SessionKeys.ANALYSIS_FILE_SIGNATURE,
        SessionKeys.DRIVE_FILES_DICT,
        SessionKeys.FAILED_DOCUMENTS,
        SessionKeys.SELECTED_DOCUMENT_ID,
        SessionKeys.SCANNING,
        SessionKeys.AUDIT_REPORT_GENERATED,
        SessionKeys.SENT_ALERTS,
        SessionKeys.WARNING_PAGE,
        "analysis_results",
        "analysis_file_signature",
        "drive_files_dict",
        "failed_documents",
        "selected_document_id",
        "scanning",
        "scanned_documents",
        "uploaded_files",
        "current_document",
        "analysis_data",
    }

    for key in list(st.session_state.keys()):
        key_str = str(key)
        if key in preserved_keys or key_str in preserved_keys:
            continue
        if (
            key in target_keys
            or key_str in target_keys
            or key_str.startswith("file_uploader")
            or key_str.startswith("doc_")
            or key_str.startswith("scan_")
            or key_str.startswith("analysis_")
        ):
            del st.session_state[key]


def reset_analysis_state() -> None:
    """Alias for reset_analysis_data() to preserve authentication state while clearing document analysis."""
    reset_analysis_data()


def reset_analysis_session_state() -> None:
    """Reset document lists, matrices, and scan flags for a new analysis.

    Keeps SessionKeys.THEME and SessionKeys.SESSION_ID.
    """
    for key in (
        SessionKeys.FAILED_DOCUMENTS,
        SessionKeys.DRIVE_FILES_DICT,
        SessionKeys.SELECTED_DOCUMENT_ID,
        SessionKeys.ANALYSIS_RESULTS,
        SessionKeys.ANALYSIS_FILE_SIGNATURE,
        SessionKeys.SCANNING,
        SessionKeys.AUDIT_REPORT_GENERATED,
        SessionKeys.SENT_ALERTS,
    ):
        if key in st.session_state:
            del st.session_state[key]

