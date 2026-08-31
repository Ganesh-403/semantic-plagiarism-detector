"""
Corpus Overview & Sidebar View Component.

Renders sidebar configuration controls, system health widget, corpus quick actions,
document management table, bulk ZIP export, and bulk clear dialogs.
"""

import logging
import os
from datetime import datetime, timezone

import psutil
import streamlit as st

from app.session_keys import SessionKeys
from app.state_manager import save_preferences_callback
from app.theme import render_timezone_footer
from src.core.config import DEFAULT_THRESHOLDS, PLAGIARISM_THRESHOLD
from src.core.document_parser import (
    DEFAULT_OCR_DPI,
    DEFAULT_OCR_LANGUAGE,
    SUPPORTED_OCR_LANGUAGES,
)
from src.db import (
    clear_all_data,
    delete_document,
    get_all_documents,
    get_unique_class_sections,
)
from src.db.auth import (
    get_tour_completed,
    get_upload_count,
    get_user_last_login,
    set_tour_completed,
)
from src.db.corpus_db import (
    get_document_char_counts,
    get_document_word_counts,
)
from src.i18n.translator import _SUPPORTED_LANGUAGES
from src.utils.bulk_export import create_documents_bulk_zip_archive
from src.utils.storage_metrics import calculate_storage_usage

try:
    from streamlit_tour import Tour
except ImportError:
    Tour = None

logger = logging.getLogger(__name__)


@st.dialog("⚠️ Confirm Bulk Clear")
def clear_all_dialog(index_path: str):
    """Dialog confirming bulk deletion of documents and database data."""
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
            if os.path.exists(index_path):
                try:
                    os.remove(index_path)
                except Exception as e:
                    logger.error(f"Error removing FAISS index: {e}")

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
            if "analysis_file_signature" in st.session_state:
                st.session_state.analysis_file_signature = None
            if "processed_pipeline_signature" in st.session_state:
                st.session_state.processed_pipeline_signature = None

            st.success("✅ All documents, chunks, and incidents have been cleared.")
            st.rerun()


@st.dialog("⚠️ Confirm Logout")
def logout_dialog(session_id: str):
    """Dialog confirming user session log out."""
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
            from src.utils.redis_cache import clear_session

            clear_session(session_id)
            st.rerun()


def render_sidebar(user_role: str, root_dir: str, faiss_index=None):
    """Render full sidebar options, health stats, threshold sliders, and filters."""
    with st.sidebar:
        if st.session_state.get(SessionKeys.AUTHENTICATED, False):
            _current_username = st.session_state.get(SessionKeys.USERNAME) or "Unknown"
            with st.sidebar.expander(f"👤 Logged in as: {_current_username}"):
                st.markdown(f"**Username:** {_current_username}")
                st.markdown(
                    f"**Role:** {user_role.capitalize() if user_role else 'N/A'}"
                )
                try:
                    _last_login = get_user_last_login(_current_username)
                except Exception:
                    _last_login = None
                st.markdown(f"**Last Login:** {_last_login if _last_login else 'N/A'}")

        try:
            total_scans_sidebar = get_upload_count()
        except Exception as e:
            logger.error(f"Failed to query total scan count for sidebar: {e}")
            total_scans_sidebar = 0

        st.markdown(f"Total Scans Processed: {total_scans_sidebar:,}")

        try:
            from src.utils.redis_cache import get_cache

            cache_inst = get_cache()
            redis_online, _ = cache_inst.ping()
            if redis_online:
                st.caption("🟢 Cache: Redis")
            else:
                st.caption("🟡 Cache: In-Memory")
        except Exception:
            st.caption("🟡 Cache: In-Memory")

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
            st.markdown("### 🎯 Threshold Presets")
            preset_options = {
                "Strict (0.80)": 0.80,
                "Balanced (0.59)": 0.59,
                "Lenient (0.45)": 0.45,
                "Custom": None,
            }

            current_threshold = st.session_state.get(
                "threshold_slider", PLAGIARISM_THRESHOLD
            )
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

            if (
                selected_preset != "Custom"
                and preset_options[selected_preset] is not None
            ):
                st.session_state["threshold_slider"] = preset_options[selected_preset]
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
                    "Recommended Default: 0.59 (59%)."
                ),
                key="threshold_slider",
                on_change=save_preferences_callback,
            )

            if abs(threshold - preset_options.get(selected_preset, -1)) > 0.001:
                if st.session_state.get("threshold_preset_radio") != "Custom":
                    st.session_state["threshold_preset_radio"] = "Custom"
                    st.rerun()

            st.slider(
                "Lexical Sensitivity Threshold",
                0.10,
                1.00,
                value=0.50,
                step=0.05,
                key=SessionKeys.LEXICAL_THRESHOLD_SLIDER,
            )

            st.slider(
                "Semantic Sensitivity Threshold",
                0.10,
                1.00,
                value=0.65,
                step=0.05,
                key=SessionKeys.SEMANTIC_THRESHOLD_SLIDER,
            )

            st.checkbox(
                "Use chunk-level similarity matrix",
                value=False,
                key=SessionKeys.CHUNK_MATRIX_CHECKBOX,
            )
            st.slider(
                "FAISS: matches per chunk",
                1,
                20,
                value=5,
                key=SessionKeys.FAISS_TOP_K_SLIDER,
            )

            from app.components.faiss_results import render_faiss_metric_badge

            render_faiss_metric_badge(st.session_state.get("faiss_index", None))

            from src.core.faiss_index import format_faiss_memory_badge

            current_faiss = faiss_index or st.session_state.get("faiss_index")
            faiss_badge_text = format_faiss_memory_badge(current_faiss)
            st.caption(f"⚡ **{faiss_badge_text}**")

            st.markdown("### ✂️ Chunking Settings")
            st.slider(
                "Chunk Size (characters)",
                200,
                2000,
                value=500,
                step=50,
                key=SessionKeys.CHUNK_SIZE_SLIDER,
            )
            st.slider(
                "Chunk Overlap (characters)",
                0,
                500,
                value=50,
                step=10,
                key=SessionKeys.CHUNK_OVERLAP_SLIDER,
            )

            with st.expander("🔤 OCR Settings", expanded=False):
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
                st.selectbox(
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
            st.button(
                "🔄 Reset All Filters",
                key="reset_all_filters_button",
                use_container_width=True,
            )

        with st.expander("⌨️ Keyboard Shortcuts"):
            st.caption("• **R**: Rerun app")
            st.caption("• **C**: Clear cache")
            st.caption("• **Tab**: Navigate focus")

        if SessionKeys.MODEL_LOAD_TIME in st.session_state:
            st.divider()
            st.caption(
                f"⚡ Vector Model Loaded in {st.session_state[SessionKeys.MODEL_LOAD_TIME]:.2f} seconds"
            )

        with st.expander("🖥️ System Health & Memory", expanded=False):
            try:
                process = psutil.Process(os.getpid())
                mem_info = process.memory_info()
                rss_mb = mem_info.rss / (1024 * 1024)

                host_limit_mb = 2048.0
                try:
                    from src.core.app_config import get_host_memory_limit_mb

                    host_limit_mb = float(get_host_memory_limit_mb())
                except Exception:
                    pass

                ram_usage_percent = min(rss_mb / host_limit_mb, 1.0)
                ram_percent_val = (rss_mb / host_limit_mb) * 100

                if ram_percent_val >= 80:
                    st.warning(
                        f"⚠️ High RAM Usage: {rss_mb:.1f} MB / {host_limit_mb:.0f} MB ({ram_percent_val:.0f}%)"
                    )
                else:
                    st.markdown(
                        f"**RAM Usage:** {rss_mb:.1f} MB / {host_limit_mb:.0f} MB ({ram_percent_val:.0f}%)"
                    )

                st.progress(ram_usage_percent)
            except Exception as mem_err:
                st.error(f"Failed to measure process memory: {mem_err}")

            st.divider()
            render_timezone_footer()

    return lang_code


def render_corpus_header(index_path: str):
    """Render top Corpus Overview title and refresh/clear header buttons."""
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
            clear_all_dialog(index_path)


def render_document_management_sidebar(
    user_role: str, index_path: str, session_id: str, last_interaction: float
):
    """Render Document Management table, bulk export ZIP, and clear buttons."""
    if user_role != "admin":
        return

    st.markdown("---")
    st.markdown("### 📁 Document Management")
    existing_docs = get_all_documents()
    if existing_docs:
        st.write(f"**{len(existing_docs)}** documents in database")

    safe_last_interaction = int(last_interaction or 0)  # noqa: F841
    st.markdown(
        """
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
        """,
        unsafe_allow_html=True,
    )

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
                        else (
                            doc.get("assignment_title")
                            if isinstance(doc, dict)
                            else None
                        )
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
                    else (
                        doc.get("assignment_title") if isinstance(doc, dict) else None
                    )
                )
                == selected_assignment
            ]

        st.write(f"**{len(existing_docs)}** documents in database")

        import pandas as pd

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
                    from src.core.faiss_index import build_index_from_matrix, save_index
                    from src.db.corpus_db import get_all_embeddings

                    embeddings_matrix = get_all_embeddings()
                    if embeddings_matrix.size > 0:
                        new_index = build_index_from_matrix(embeddings_matrix)
                        save_index(new_index, index_path)
                    else:
                        if os.path.exists(index_path):
                            os.remove(index_path)
                    st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button(
        "🗑️ Clear All Documents",
        key="clear_all_documents_button",
        use_container_width=True,
    ):
        clear_all_dialog(index_path)
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🚪 Log Out", use_container_width=True, key="logout_button"):
        logout_dialog(session_id)


def render_onboarding_tour(user_role: str):
    """Render admin onboarding tour if not completed."""
    if (
        Tour is not None
        and user_role == "admin"
        and not get_tour_completed(st.session_state.get(SessionKeys.USERNAME, ""))
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
