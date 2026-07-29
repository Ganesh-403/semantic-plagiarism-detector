import asyncio
import base64
import html
import io as _io
import logging
import os
from pathlib import Path
import sys
import time
from datetime import datetime
from typing import Any

# Fix Streamlit import paths by pointing to project root
FILE_PATH = Path(__file__).resolve()
ROOT_DIR = FILE_PATH.parent.parent  # Points to semantic-plagiarism-detector/

# Silence harmless Windows asyncio Proactor connection lost bugs
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import streamlit as st

from app.theme import (
    back_to_top_html,
    empty_state_html,
    get_colors,
    get_theme_name,
    inject_css,
    set_theme,
)
from src.core.ai_detector import detect_documents_ai_probability
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
from src.core.embedding_model import embed_chunks, embed_documents
from src.core.export_engine import LMSExportEngine
from src.core.faiss_index import (
    build_index,
    build_index_from_matrix,
    load_or_rebuild_index,
    save_index,
    search_similar_chunks,
)
from src.core.similarity import (
    document_similarity_matrix,
    find_most_similar_chunks,
    flag_plagiarism,
)
from src.core.webhook import send_plagiarism_alert
from src.db.auth import (
    authenticate_user,
    disable_2fa,
    enable_2fa,
    get_2fa_status,
    get_all_users,
    get_notification_preferences,
    get_tour_completed,
    get_user_preferences,
    init_db,
    is_user_active,
    set_tour_completed,
    set_user_active_status,
    update_notification_preferences,
)
from src.db.corpus_db import (
    delete_document,
    get_all_documents,
    get_all_embeddings,
    get_chunk_registry,
    get_unique_class_sections,
    init_corpus_db,
)
from src.db.incidents import (
    get_all_incidents_above_threshold_for_export,
    get_high_severity_trends,
    get_most_plagiarized_documents,
    sync_flagged_incidents,
)
from src.i18n.translator import _SUPPORTED_LANGUAGES, get_text
from src.security.metadata_stripper import strip_exif_metadata
from src.utils.diff_highlighter import highlight_overlap
from src.utils.excel_export import export_similarity_matrix_to_excel
from src.utils.filename import (
    InvalidFileExtensionError,
    sanitize_filename,
    unique_filename,
    validate_document_extension,
)
from src.utils.json_export import export_similarity_matrix_to_json
from src.utils.redis_cache import (
    cache_analysis_results,
    cache_faiss_index,
    cache_session_state,
    clear_session,
    get_analysis_results,
    get_faiss_index,
    get_session_state,
)

logger = logging.getLogger(__name__)

# Initialize databases
init_corpus_db()
init_db()

# Safe import for PDF Highlighting
try:
    from src.utils.pdf_highlighter import highlight_pdf_matches
except Exception:
    highlight_pdf_matches = None

try:
    from streamlit_tour import Tour
except ImportError:
    Tour = None

# Initialize unique session ID
if "session_id" not in st.session_state:
    import uuid

    st.session_state.session_id = str(uuid.uuid4())

SESSION_ID = st.session_state.session_id
_INDEX_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "corpus.index")
)

# Page Configuration
st.set_page_config(
    page_title="Semantic Plagiarism Detector",
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

st.markdown(back_to_top_html(), unsafe_allow_html=True)
inject_css()

# ── SESSION TIMEOUT & ROUTE PROTECTION ────────────────────────────────────────
TIMEOUT_LIMIT = 15 * 60  # 15 minutes

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
        st.warning("⚠️ Session expired due to inactivity. Please log in again.")
        st.stop()
    else:
        st.session_state.last_interaction = time.time()
        cache_session_state(SESSION_ID, "last_interaction", time.time())

# Render Login UI if not authenticated
if not st.session_state.get("authenticated", False):
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

# Active user role
user_role = st.session_state.get("role", "admin")

# ── Top-right Theme Toggle ───────────────────────────────────────────────────
current_theme = get_theme_name()
_, theme_col = st.columns([0.94, 0.06])

with theme_col:
    theme_icon = "☀️" if current_theme == "Dark" else "🌙"
    if st.button(theme_icon, key="theme_toggle"):
        new_theme = "Light" if current_theme == "Dark" else "Dark"
        set_theme(new_theme)
        st.rerun()

# ── Sidebar (ROLE RESTRICTED Settings) ────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    if user_role == "admin":
        st.markdown("### ⚙️ Sensitivity & Thresholds")

        lexical_threshold = st.slider(
            "Lexical Similarity Threshold",
            0.0,
            1.0,
            value=st.session_state.get("lexical_threshold_slider", 0.70),
            step=0.01,
            help=(
                "Measures direct word-for-word and string overlap. "
                "Higher values require near-exact phrasing to flag matches. "
                "Recommended default: 0.70 (70%)."
            ),
            key="lexical_threshold_slider",
        )

        semantic_threshold = st.slider(
            "Semantic Similarity Threshold",
            0.0,
            1.0,
            value=st.session_state.get("semantic_threshold_slider", 0.75),
            step=0.01,
            help=(
                "Measures underlying meaning and contextual similarity using embedding vectors. "
                "Detects paraphrased or restructured content even if words differ. "
                "Recommended default: 0.75 (75%)."
            ),
            key="semantic_threshold_slider",
        )

        threshold = st.slider(
            "Hybrid Similarity Threshold",
            0.0,
            1.0,
            value=st.session_state.get("threshold_slider", PLAGIARISM_THRESHOLD),
            step=0.01,
            help=(
                "Combined similarity score blending both lexical and semantic metrics. "
                "Pairs scoring above this threshold are flagged for plagiarism review. "
                "Recommended default: 0.80 (80%)."
            ),
            key="threshold_slider",
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
    else:
        lexical_threshold = 0.70
        semantic_threshold = 0.75
        threshold = PLAGIARISM_THRESHOLD
        use_chunk_matrix = False
        faiss_top_k = 5
        chunk_size = 500
        chunk_overlap = 50
        ocr_language = DEFAULT_OCR_LANGUAGE
        ocr_dpi = DEFAULT_OCR_DPI

    unique_classes = ["All Classes"] + get_unique_class_sections()
    selected_class = st.selectbox(
        "Select Class/Section",
        unique_classes,
        index=0,
        key="class_filter_selectbox",
    )

    st.markdown("---")
    st.markdown(f"👤 Logged in as **{st.session_state.get('username', '')}**")
    if st.button("🚪 Log Out", use_container_width=True, key="logout_button"):
        for key in ["authenticated", "username", "role", "last_interaction"]:
            if key in st.session_state:
                del st.session_state[key]
        clear_session(SESSION_ID)
        st.rerun()

# ── Main UI ───────────────────────────────────────────────────────────────────
st.title("🔍 Semantic Plagiarism Detection System")

uploaded_files = st.file_uploader(
    "📂 Upload Assignments",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
    key="file_uploader",
)

file_bytes_dict = (
    {f.name: f.getvalue() for f in uploaded_files} if uploaded_files else {}
)

if len(file_bytes_dict) < 2:
    st.info("Upload at least 2 files to begin analysis.")
    st.stop()


# ── Pipeline Execution ────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_pipeline(
    file_bytes_dict: dict[str, bytes],
    ocr_language: str,
    ocr_dpi: int,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
):
    raw_texts = {}
    for name, data in file_bytes_dict.items():
        raw_texts[name] = extract_text(
            _io.BytesIO(data),
            name,
            ocr_language=ocr_language,
            ocr_dpi=ocr_dpi,
        )

    chunked_docs = {}
    for doc_name, text in raw_texts.items():
        chunked_docs[doc_name] = [
            text[i : i + chunk_size]
            for i in range(0, len(text), chunk_size - chunk_overlap)
        ]

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
                    float(np.max(np.dot(ea, eb.T))) if ea.size and eb.size else 0.0
                )
                chunk_mat[i, j] = score
                chunk_mat[j, i] = score

    chunk_sim_df = pd.DataFrame(chunk_mat, index=names, columns=names)
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

st.dataframe(
    active_sim_df.style.background_gradient(cmap="OrRd").format("{:.4f}"),
    use_container_width=True,
)
