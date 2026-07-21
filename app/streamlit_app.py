import sys
import os

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import hashlib  # noqa: E402
import io as _io  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from app.theme import (  # noqa: E402
    get_theme_name,
    inject_css,
    set_theme,
)
from sklearn.metrics.pairwise import cosine_similarity  # noqa: E402
from src.core.document_parser import (  # noqa: E402
    OCRDependencyError,
    extract_text,
)
from src.core.embedding_model import (  # noqa: E402
    embed_documents,
    get_embedding_model_info,
)
from src.core.faiss_index import (  # noqa: E402
    build_index,
    build_index_from_matrix,
    find_plagiarised_chunks,
    load_index,
    save_index,
    search_similar_chunks,
)
from src.core.similarity import (  # noqa: E402
    PLAGIARISM_THRESHOLD,
    document_similarity_matrix,
    find_most_similar_chunks,
    flag_plagiarism,
)
from src.core.text_chunking import chunk_documents  # noqa: E402
from src.core.webhook import send_plagiarism_alert  # noqa: E402
from src.db import (  # noqa: E402
    add_chunks,
    add_document,
    delete_document,
    get_all_documents,
    get_all_embeddings,
    get_chunk_registry,
    get_document_by_hash,
    get_documents_by_class,
    get_unique_class_sections,
    init_corpus_db,
)
from src.db.auth import (  # noqa: E402
    add_user,
    delete_user,
    get_all_users,
    get_user_role,
    init_db,
    update_password,
    verify_user,
)
from src.utils.pdf_report import generate_plagiarism_report  # noqa: E402
from src.visualization.heatmap import (  # noqa: E402
    plot_chunk_similarity_comparison,
    plot_similarity_heatmap,
)
from src.visualization.network_graph import plot_similarity_network  # noqa: E402
from typing import Any  # noqa: E402
from utils.warning_list import render_warning_controls  # noqa: E402

# Initialize corpus database
init_corpus_db()

# FAISS index file path
_INDEX_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "corpus.index"))

# Branding config persistence
_BRANDING_CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "branding_config.json"))
_BRANDING_LOGO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "branding_logo.png"))

# Initialize database
init_db()
# Must be the first Streamlit command called
st.set_page_config(
    page_title="Semantic Plagiarism Detector",
    page_icon="🔍", layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()
init_db()
st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    .stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ── SESSION TIMEOUT & ROUTE PROTECTION MIDDLEWARE ─────────────────────────────
TIMEOUT_LIMIT = 15 * 60  # 15 minutes in seconds

# 1. Handle Automatic Session Expiration (Inactivity Check)
if "last_interaction" in st.session_state and st.session_state.get("authenticated", False):
    elapsed_time = time.time() - st.session_state.last_interaction
    if elapsed_time > TIMEOUT_LIMIT:
        # Clear sensitive session variables on timeout
        for key in [
            "authenticated",
            "username",
            "role",
            "last_interaction",
            "analysis_results",
            "analysis_file_signature",
        ]:
            if key in st.session_state:
                del st.session_state[key]
        st.warning("⏱️ Your session has expired due to 15 minutes of inactivity. Please log in again.")
        st.stop()
    else:
        # Record a timestamp of the latest user interaction
        st.session_state.last_interaction = time.time()

# 2. Render Login UI if not authenticated
if not st.session_state.get("authenticated", False):
    st.title("🔒 Plagiarism Detection Portal Login")
    st.markdown("Please log in with your credentials to access the system.")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login_submitted = st.form_submit_button("Log In", use_container_width=True)
        
        if login_submitted:
            if verify_user(username, password):
                role = get_user_role(username)
                st.session_state.authenticated = True
                st.session_state.role = role
                st.session_state.username = username
                st.session_state.last_interaction = time.time()
                st.success(f"Welcome back, {role.capitalize()}!")
                st.rerun()
            username = username.strip().lower()

            if not username or not password:
                st.error("Please enter both username and password.")

            elif verify_user(username, password):
                role = get_user_role(username)

                if role is None:
                    st.error("Unable to determine the user role.")
                else:
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.session_state.role = role
                    st.session_state.last_interaction = time.time()

                    st.success(f"Welcome back, {username}!")
                    st.rerun()

            else:
                st.error("Invalid username or password.")
    st.stop()

# Get secure role for this active interaction
user_role = st.session_state.get("role", "user")

model_name, embedding_dim = get_embedding_model_info()

# ── Sidebar (ROLE RESTRICTED Settings) ────────────────────────────────────────
with st.sidebar:
    st.markdown("<div style='font-size: 72px; line-height: 1;'>🕵️‍♂️</div>", unsafe_allow_html=True)
    st.title("⚙️ Settings")
    
    current_theme = get_theme_name()

    selected_theme = st.radio(
    "Theme",
    options=["Light", "Dark"],
    index=0 if current_theme == "Light" else 1,
    horizontal=True,
    key="theme_selector",
)
    if selected_theme != current_theme:
      set_theme(selected_theme)
      st.rerun()
    st.write(f"Logged in as: **{user_role.upper()}**")

    # Only show administrative settings to ADMIN users
    if user_role == "admin":
        threshold = st.slider("Plagiarism Threshold", 0.50, 0.99,
                              value=PLAGIARISM_THRESHOLD, step=0.01,
                              help="Cosine similarity above which a pair is flagged. (Recommended: 0.59 based on benchmark evaluation)")
        use_chunk_matrix = st.checkbox("Use chunk-level similarity matrix", value=False,
                                       help="Use MAX chunk-pair similarity instead of mean doc vectors.")
        faiss_top_k = st.slider("FAISS: matches per chunk", 1, 20, value=5,
                                help="Nearest neighbours per chunk in FAISS search.")
    else:
        # Fallbacks for Standard Users (Cannot alter thresholds or configs)
        threshold = PLAGIARISM_THRESHOLD
        use_chunk_matrix = False
        faiss_top_k = 5
        st.info("ℹ️ Settings configuration is restricted to Administrators.")

    st.markdown("---")
    st.markdown("### 🔍 Class Filter")
    unique_classes = ["All Classes"] + get_unique_class_sections()
    selected_class = st.selectbox(
        "Select Class/Section",
        unique_classes,
        index=0,
        help="Filter the analysis dashboard and similarity matrices by a specific class section."
    )

    # ── Report Branding (Admin only) ───────────────────────────────────────
    if user_role == "admin":
        st.markdown("---")
        with st.expander("🎨 Report Branding", expanded=False):
            st.caption("Customize PDF report appearance with your institution branding.")

            def _load_branding_config() -> dict:
                defaults = {"brand_color": "#1e3a8a", "logo_path": None}
                try:
                    if os.path.exists(_BRANDING_CONFIG_PATH):
                        with open(_BRANDING_CONFIG_PATH, "r") as f:
                            cfg = json.load(f)
                        defaults.update(cfg)
                except (json.JSONDecodeError, OSError):
                    pass
                return defaults

            def _save_branding_config(cfg: dict) -> None:
                with open(_BRANDING_CONFIG_PATH, "w") as f:
                    json.dump(cfg, f, indent=2)

            _branding_cfg = _load_branding_config()

            brand_color = st.color_picker(
                "Brand Color",
                value=_branding_cfg.get("brand_color", "#1e3a8a"),
                help="Applied to all PDF headers and section titles.",
            )

            logo_file = st.file_uploader(
                "Institution Logo",
                type=["png", "jpg", "jpeg"],
                help="Logo appears in the PDF report header. Recommended: transparent PNG, max 300px wide.",
            )

            if logo_file is not None:
                try:
                    from PIL import Image as _PILImage
                    import io as _img_io

                    _img_bytes = logo_file.read()
                    _img = _PILImage.open(_img_io.BytesIO(_img_bytes))
                    max_w = 300
                    if _img.width > max_w:
                        ratio = max_w / _img.width
                        _img = _img.resize((max_w, int(_img.height * ratio)), _PILImage.LANCZOS)
                    buf = _img_io.BytesIO()
                    _img.save(buf, format="PNG")
                    logo_bytes = buf.getvalue()
                    st.image(logo_bytes, width=150, caption="Logo preview")
                    with open(_BRANDING_LOGO_PATH, "wb") as f:
                        f.write(logo_bytes)
                except Exception:
                    logo_bytes = logo_file.getvalue()
                    st.image(logo_bytes, width=150, caption="Logo preview")
                    with open(_BRANDING_LOGO_PATH, "wb") as f:
                        f.write(logo_bytes)
                _branding_cfg["logo_path"] = _BRANDING_LOGO_PATH
            elif os.path.exists(_BRANDING_LOGO_PATH):
                _branding_cfg["logo_path"] = _BRANDING_LOGO_PATH
            else:
                _branding_cfg["logo_path"] = None

            _branding_cfg["brand_color"] = brand_color
            _save_branding_config(_branding_cfg)

            st.session_state["brand_color"] = brand_color
            st.session_state["logo_path"] = _branding_cfg.get("logo_path")

    st.markdown("---")
    st.markdown("""
**How it works**
1. Upload **PDF, DOCX, or TXT** assignment files
2. Text is extracted according to the file type
3. Text is split into **paragraph chunks**
4. Chunks are embedded with **{model_name}**
5. A **FAISS index** is built over all chunk vectors
6. Pairs above the threshold are flagged
""")
    st.info(
    f"**Active Model:** {model_name}\n\n"
    f"**Dimension:** {embedding_dim}"
)
    st.markdown("---")
    st.caption("Semantic Plagiarism Detector · FAISS edition")
    
    # Log out button
    if st.button("🚪 Log Out", use_container_width=True):
        for key in ["authenticated", "username", "role", "last_interaction"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🔍 Semantic Plagiarism Detection System")
st.markdown(
    "Upload student PDF, DOCX, or TXT files. Detects **semantic similarity** "
    "(even paraphrased text) using transformer embeddings + "
    "**FAISS vector search**."
)
st.divider()

# ── MAIN APPLICATION SECTIONS (ROLE CHECKED) ──────────────────────────────────

if user_role != "admin":
    # STANDARD USER VIEW: Student Query / Search Panel Only (No admin PDF uploading)
    st.subheader("🔎 Secure Student Search Portal")
    st.caption("Paste a text snippet below to check its similarity against existing indexed assignments.")
    
    st.info("🔒 Note: Direct assignment uploads and detailed breakdown panels are restricted to Administrator access. Your queries are anonymized for privacy.")
    
    query_text = st.text_area("Paste a text snippet to check against index:", height=150,
                              placeholder="Paste a paragraph here to check for plagiarism...")
    
    if st.button("🔍 Run Quick Verification", key="user_query") and query_text.strip():
        # Load existing index and registry from database
        from src.db.corpus_db import get_chunk_registry, get_all_embeddings
        from src.core.faiss_index import build_index_from_matrix
        
        with st.spinner("Loading index and searching..."):
            try:
                registry = get_chunk_registry()
                embeddings_matrix = get_all_embeddings()
                
                if embeddings_matrix.shape[0] == 0:
                    st.warning("No documents are currently indexed. Please contact your administrator.")
                else:
                    # Build index from stored embeddings
                    faiss_index = build_index_from_matrix(embeddings_matrix, index_type="auto")
                    
                    # Embed the query
                    from src.core.embedding_model import embed_chunks
                    query_vec = embed_chunks([query_text.strip()])[0]
                    
                    # Search with threshold
                    faiss_threshold = threshold
                    results = search_similar_chunks(query_vec, faiss_index, registry,
                                                      top_k=faiss_top_k, threshold=faiss_threshold)
                    
                    if selected_class != "All Classes":
                        class_docs = get_documents_by_class(selected_class)
                        results = [(record, score) for record, score in results if record.doc_name in class_docs]
                    
                    if not results:
                        st.success("✅ No significant matches found in the assignment database.")
                    else:
                        st.success(f"Found **{len(results)}** potentially similar passages.")
                        
                        # Anonymize document names
                        doc_id_map = {}
                        anon_counter = 1
                        
                        for record, score in results:
                            if record.doc_name not in doc_id_map:
                                doc_id_map[record.doc_name] = f"Document-{anon_counter:03d}"
                                anon_counter += 1
                        
                        # Display anonymized results
                        for rank, (record, score) in enumerate(results, 1):
                            anon_doc_name = doc_id_map[record.doc_name]
                            color = "#ff4b4b" if score >= 0.90 else "#ffa500"
                            
                            with st.expander(
                                f"#{rank} · {anon_doc_name} (chunk #{record.chunk_index+1}) "
                                f"— {score:.1%}",
                                expanded=(rank == 1)
                            ):
                                cq, cm = st.columns(2)
                                with cq:
                                    st.markdown("**Your query:**")
                                    st.info(query_text.strip())
                                with cm:
                                    st.markdown(f"**Matching passage in {anon_doc_name}:**")
                                    st.warning(record.chunk_text)
                                
                                st.markdown(
                                    f"<div style='text-align:right;'>"
                                    f"<span style='background:{color};color:white;padding:3px 12px;"
                                    f"border-radius:10px;font-size:0.85rem;font-weight:700;'>"
                                    f"Similarity: {score*100:.1f}%</span></div>",
                                    unsafe_allow_html=True,
                                )
                        
                        st.caption("🔒 Document names are anonymized to protect student privacy.")
                        
            except Exception as e:
                st.error(f"Error loading index: {str(e)}")
                st.info("Please ensure documents have been indexed by an administrator.")
else:
    # ADMINISTRATOR ACCESS: Full Upload Pipeline & Evaluation Dashboards
    
    # Load or initialize FAISS index
    if os.path.exists(_INDEX_PATH):
        faiss_index = load_index(_INDEX_PATH)
        registry = get_chunk_registry()
        st.info(f"📂 Loaded existing FAISS index with {faiss_index.ntotal} vectors")
    else:
        faiss_index = None
        registry = []

    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = None

    if "analysis_file_signature" not in st.session_state:
        st.session_state.analysis_file_signature = None
    
    uploaded_files = st.file_uploader(
        "📂 Upload Assignment PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload 2 or more PDF files.",
    )

    
    # Use getvalue() so uploaded bytes remain available on every Streamlit rerun.
    # Calling read() advances the UploadedFile stream and can return empty bytes
    # after changing a tab, selectbox, checkbox, or slider.
    file_bytes_dict = {
        uploaded_file.name: uploaded_file.getvalue()
        for uploaded_file in uploaded_files
    }
    
    # Allow analysis with existing index even without new uploads
    if not uploaded_files or len(uploaded_files) < 2:
        if st.session_state.analysis_results is None:
            st.info("👆 Please upload **at least 2** PDF assignment files to begin.")
            st.stop()
        else:
            st.success(f"📂 Using existing index with {faiss_index.ntotal} vectors from {len(get_all_documents())} documents")
            # Skip to analysis section with existing index
            file_bytes_dict = {}
            raw_texts = {}
            chunked_docs = {}
            embeddings = {}
            sim_df = None
            chunk_sim_df = None
            # We'll need to handle this case differently for the analysis
            st.warning("⚠️ Full similarity matrix requires re-uploading files. FAISS search is available with existing index.")
            # For now, require uploads for full functionality
            st.stop()
    
    if len(uploaded_files) < 2:
        st.info("👆 Please upload **at least 2** PDF, DOCX, or TXT assignment files to begin.")
        st.stop()

    # ── Metadata Editor Section ──────────────────────────────────────────────────
    st.markdown("### 📝 Set Document Metadata")
    
    col1, col2 = st.columns(2)
    with col1:
        batch_class = st.text_input("Default Class/Section", value="Class A", help="Default class section for all files in this batch.")
    with col2:
        batch_assignment = st.text_input("Default Assignment Title", value="Assignment 1", help="Default assignment title for all files in this batch.")
        
    st.markdown("Customize metadata for individual files if needed:")
    metadata_dict = {}
    for f in uploaded_files:
        base_name = os.path.splitext(f.name)[0]
        guessed_name = base_name.replace("_", " ").replace("-", " ").title()
        
        with st.expander(f"📄 {f.name}", expanded=False):
            student_name = st.text_input(f"Student Name for {f.name}", value=guessed_name, key=f"student_{f.name}")
            class_section = st.text_input(f"Class/Section for {f.name}", value=batch_class, key=f"class_{f.name}")
            assignment_title = st.text_input(f"Assignment Title for {f.name}", value=batch_assignment, key=f"assignment_{f.name}")
            
            metadata_dict[f.name] = {
                "student_name": student_name.strip(),
                "class_section": class_section.strip(),
                "assignment_title": assignment_title.strip()
            }

    # ── OCR failure UI ─────────────────────────────────────────────────────────────
    class OCRFileBatchError(OCRDependencyError):
        """OCR dependency failure containing the names of affected PDF files."""

        def __init__(
            self,
            failed_files: list[str],
            details: list[str],
        ) -> None:
            self.failed_files = failed_files
            self.details = details
            super().__init__("; ".join(details))


    def show_ocr_dependency_error(
        failed_files: list[str],
        error_message: str,
    ) -> None:
        """Render a user-friendly OCR dependency and installation guide."""
        st.error("⚠️ OCR could not process one or more scanned PDF files.")

        with st.expander("View OCR setup instructions", expanded=True):
            st.markdown(
                """
                The affected PDF files appear to contain scanned or image-based
                pages. These pages require **Tesseract OCR**, but Tesseract or
                one of the required Python OCR packages is unavailable.
                """
            )

            st.markdown("#### Files that could not be processed")
            for filename in failed_files:
                st.markdown(f"- `{filename}`")

            st.markdown("#### Install Tesseract OCR")
            st.markdown(
                """
                **Windows**

                1. Install Tesseract OCR.
                2. Add the Tesseract installation directory to your system
                   `PATH`.
                3. Alternatively, set `TESSERACT_CMD` to the full path of
                   `tesseract.exe`.

                **macOS**

                ```bash
                brew install tesseract
                ```

                **Ubuntu/Debian Linux**

                ```bash
                sudo apt update
                sudo apt install tesseract-ocr
                ```

                **Required Python packages**

                ```bash
                python -m pip install pytesseract pymupdf pillow
                ```

                Restart the Streamlit application after installation.
                """
            )

            st.markdown("### Technical details")
            st.code(error_message)


    # ── Pipeline (cached) ─────────────────────────────────────────────────────────
    @st.cache_data(show_spinner=False)
    def run_pipeline(
        file_bytes_dict: dict[str, bytes],
        existing_index=None,
        existing_registry=None,
    ):
        """Extract, chunk, embed and index uploaded assignment files."""
        raw_texts = {}
        failed_files = []
        failure_details = []

        # Process every uploaded file and collect OCR failures from scanned PDFs.
        for name, data in file_bytes_dict.items():
            try:
                raw_texts[name] = extract_text(_io.BytesIO(data), name)
            except OCRDependencyError as exc:
                failed_files.append(name)
                failure_details.append(f"{name}: {exc}")

        if failed_files:
            raise OCRFileBatchError(failed_files, failure_details)

        # Original chunks are preserved for UI display.
        chunked_docs = chunk_documents(raw_texts)

        # Translated English chunks are used only for embeddings.
        from src.core.cross_lingual import prepare_documents_for_embedding

        translated_chunked_docs, alignment_metadata = (
            prepare_documents_for_embedding(chunked_docs)
        )

        # Generate embeddings from translated English text.
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
                    ea = embeddings[na]
                    eb = embeddings[nb]

                    score = (
                        float(np.max(cosine_similarity(ea, eb)))
                        if ea.size and eb.size
                        else 0.0
                    )

                    chunk_mat[i, j] = score
                    chunk_mat[j, i] = score

        chunk_sim_df = pd.DataFrame(
            chunk_mat,
            index=names,
            columns=names,
        )

        # Original chunks remain in the registry, while vectors are generated
        # from translated English text.
        faiss_index, registry = build_index(
            embeddings,
            chunked_docs,
        )

        return (
            raw_texts,
            chunked_docs,
            embeddings,
            sim_df,
            chunk_sim_df,
            faiss_index,
            registry,
        )


    # ── Persistent analysis state ────────────────────────────────────────────────
    # Streamlit reruns the full script whenever a widget changes. Keep the
    # completed pipeline outputs in session_state so tab and selectbox changes do
    # not reset document counts, similarity matrices, flags, or FAISS metrics.
    file_signature = tuple(
        sorted(
            (
                name,
                len(data),
                hashlib.sha256(data).hexdigest(),
            )
            for name, data in file_bytes_dict.items()
        )
    )

    analysis_is_current = (
        st.session_state.analysis_results is not None
        and st.session_state.analysis_file_signature == file_signature
    )

    if not analysis_is_current:
        try:
            with st.spinner(
                "🧠 Processing PDFs, building embeddings and FAISS index…"
            ):
                analysis_results = run_pipeline(file_bytes_dict)
        except OCRDependencyError as exc:
            failed_files = getattr(
                exc,
                "failed_files",
                list(file_bytes_dict.keys()),
            )
            show_ocr_dependency_error(
                failed_files=failed_files,
                error_message=str(exc),
            )
            st.stop()

        (
            raw_texts,
            chunked_docs,
            embeddings,
            sim_df,
            chunk_sim_df,
            faiss_index,
            registry,
        ) = analysis_results

        st.session_state.analysis_results = analysis_results
        st.session_state.analysis_file_signature = file_signature

        # Persist only documents that are not already stored. Database duplicate
        # detection must not decide whether dashboard data remains visible.
        saved_documents = 0
        skipped_documents = []

        for uploaded_file in uploaded_files:
            file_data = file_bytes_dict[uploaded_file.name]
            file_hash = hashlib.sha256(file_data).hexdigest()
            existing = get_document_by_hash(file_hash)

            if existing:
                skipped_documents.append(uploaded_file.name)
                continue

            meta = metadata_dict.get(
                uploaded_file.name,
                {
                    "student_name": "",
                    "class_section": "",
                    "assignment_title": "",
                },
            )

            add_document(
                uploaded_file.name,
                file_hash,
                class_section=meta["class_section"],
                student_name=meta["student_name"],
                assignment_title=meta["assignment_title"],
            )

            doc_embeddings = embeddings.get(uploaded_file.name)
            doc_chunks = chunked_docs.get(uploaded_file.name, [])

            if (
                doc_embeddings is not None
                and getattr(doc_embeddings, "ndim", 0) == 2
                and doc_embeddings.shape[0] > 0
            ):
                chunks_to_add = []

                # The database stores a globally unique vector/index identifier.
                existing_registry_size = len(get_chunk_registry())
                for chunk_index, (vector, chunk_text) in enumerate(
                    zip(doc_embeddings, doc_chunks)
                ):
                    chunks_to_add.append(
                        (
                            existing_registry_size + chunk_index,
                            uploaded_file.name,
                            chunk_index,
                            chunk_text,
                            vector,
                        )
                    )

                if chunks_to_add:
                    add_chunks(chunks_to_add)

            saved_documents += 1

        if skipped_documents:
            st.info(
                f"⏭️ Already stored in the database: "
                f"{', '.join(skipped_documents)}"
            )

        if saved_documents:
            st.success(
                f"✅ Added {saved_documents} new document"
                f"{'s' if saved_documents != 1 else ''} to the database."
            )

        # Save the current analysis index. The in-memory analysis remains the
        # source of truth for this upload set throughout the Streamlit session.
        if faiss_index is not None:
            save_index(faiss_index, _INDEX_PATH)

    else:
        (
            raw_texts,
            chunked_docs,
            embeddings,
            sim_df,
            chunk_sim_df,
            faiss_index,
            registry,
        ) = st.session_state.analysis_results

    # Optional explicit reset. Normal widget changes must never clear analysis.
    if st.button(
        "🗑️ Clear current analysis",
        key="clear_current_analysis",
        help="Clear the current upload analysis and start with a new set of files.",
    ):
        st.session_state.analysis_results = None
        st.session_state.analysis_file_signature = None
        run_pipeline.clear()
        st.rerun()

    empty_docs = [
        name
        for name, extracted_text in raw_texts.items()
        if not extracted_text.strip()
    ]
    if empty_docs:
        st.warning(
            f"⚠️ **Could not extract text from:** {', '.join(empty_docs)}. "
            "The files may be empty, unsupported, scanned, corrupted, "
            "or password-protected."
        )

    active_sim_df = chunk_sim_df if use_chunk_matrix else sim_df
    flags = (
        flag_plagiarism(active_sim_df, threshold=threshold)
        if active_sim_df is not None
        else []
    )

    # Apply Class/Section filter if selected
    if selected_class != "All Classes":
        class_docs = get_documents_by_class(selected_class)
        # Filter raw_texts, chunked_docs, embeddings
        raw_texts = {k: v for k, v in raw_texts.items() if k in class_docs}
        chunked_docs = {k: v for k, v in chunked_docs.items() if k in class_docs}
        embeddings = {k: v for k, v in embeddings.items() if k in class_docs}
        
        # Slice active_sim_df if available
        if active_sim_df is not None:
            filtered_names = [name for name in active_sim_df.index if name in class_docs]
            if len(filtered_names) >= 2:
                active_sim_df = active_sim_df.loc[filtered_names, filtered_names]
                # Re-calculate plagiarism flags for the filtered subset
                flags = flag_plagiarism(active_sim_df, threshold=threshold)
            else:
                active_sim_df = None
                flags = []
                st.warning(f"⚠️ Need at least 2 documents in '{selected_class}' to display the similarity matrix and warnings.")

    # ── Webhook notifications for high-similarity matches (>= 90%) ───────────────
    if "notified_pairs" not in st.session_state:
        st.session_state.notified_pairs = set()
    
    if active_sim_df is not None:
        # Use the currently analysed upload set. The previous implementation
        # referenced `new_files`, but that variable no longer exists after the
        # session-persistence refactor.
        current_files = sorted(raw_texts.keys())
        if (
            "last_uploaded_files" not in st.session_state
            or st.session_state.last_uploaded_files != current_files
        ):
            st.session_state.last_uploaded_files = current_files
            st.session_state.notified_pairs = set()

        for flag in flags:
            if flag["similarity"] >= 0.90:
                pair_key = tuple(sorted([flag["doc_a"], flag["doc_b"]]))
                if pair_key not in st.session_state.notified_pairs:
                    send_plagiarism_alert(flag["doc_a"], flag["doc_b"], flag["similarity"])
                    st.session_state.notified_pairs.add(pair_key)

    # ── Summary metrics ───────────────────────────────────────────────────────────
    st.subheader("📊 Analysis Summary")
    col1, col2, col3, col4, col5 = st.columns(5)
    doc_names    = list(raw_texts.keys())
    n_docs       = len(doc_names)
    total_pairs  = n_docs * (n_docs - 1) // 2 if n_docs > 1 else 0
    n_flagged    = len(flags)
    n_high       = sum(1 for f in flags if "High" in f["severity"])
    avg_sim      = active_sim_df.values[np.triu_indices(n_docs, k=1)].mean() if active_sim_df is not None and n_docs > 1 else 0.0
    total_chunks = sum(len(v) for v in chunked_docs.values())

    col1.metric("📄 Documents",   n_docs)
    col2.metric("🔗 Pairs",       total_pairs)
    col3.metric("🚨 Flagged",     n_flagged,
                delta=f"{n_high} High" if n_high else None, delta_color="inverse")
    col4.metric("📈 Avg Similarity", f"{avg_sim:.1%}")
    col5.metric(
        "🗂️ FAISS Vectors",
        faiss_index.ntotal if faiss_index is not None else 0,
    )
    st.divider()

    # ── Tabs ──────────────────────────────────────────────────────────────────────
    (
        tab_warnings, tab_faiss, tab_matrix, tab_heatmap,
        tab_drill, tab_corpus, tab_users,
    ) = st.tabs([
        "⚠️ Plagiarism Warnings",
        "⚡ FAISS Chunk Search",
        "📋 Similarity Matrix",
        "🗺️ Heatmap",
        "🔬 Pair Drill-Down",
        "🗃️ Manage Corpus",
        "👥 User Management",
    ])

    # ══ TAB 1 ════════════════════════════════════════════════════════════════════
    with tab_warnings:
        st.subheader("⚠️ Plagiarism Warnings")
        render_warning_controls(flags, threshold=threshold)

    # ══ TAB 2: FAISS ═════════════════════════════════════════════════════════════
    with tab_faiss:
        st.subheader("⚡ FAISS Vector Search — Chunk-Level Plagiarism")
        st.markdown(
            "FAISS searches **every chunk** against all other documents' chunks. "
            "Uses exact search for small collections and **IVF approximate search** "
            "for large ones — scaling to thousands of assignments."
        )

        faiss_col1, faiss_col2 = st.columns([2, 1])
        with faiss_col1:
            faiss_threshold = st.slider("FAISS similarity threshold", 0.50, 0.99,
                                        value=threshold, step=0.01, key="faiss_thresh")
        with faiss_col2:
            run_faiss = st.button("🔍 Run FAISS Search", type="primary", use_container_width=True)

        st.info(f"📐 Index: **{faiss_index.ntotal} vectors** across **{n_docs} documents** "
                f"({total_chunks} chunks total).")

        if run_faiss:
            with st.spinner("Searching FAISS index across all chunks…"):
                faiss_matches = find_plagiarised_chunks(
                    embeddings, chunked_docs, faiss_index, registry,
                    threshold=faiss_threshold, top_k=faiss_top_k,
                )
            
            if selected_class != "All Classes":
                faiss_matches = [m for m in faiss_matches if m["match_doc"] in class_docs]

            if not faiss_matches:
                st.success("✅ No chunk-level matches found above the threshold.")
            else:
                st.success(f"Found **{len(faiss_matches)} suspicious chunk pairs**.")
                summary_rows = [{
                    "Source Document": m["source_doc"],
                    "Matched Document": m["match_doc"],
                    "Similarity": f"{m['similarity']:.1%}",
                    "Severity": "🔴 High" if m["similarity"] >= 0.90 else "🟡 Medium",
                } for m in faiss_matches]
                st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

                st.subheader("🔑 Matching Paragraph Pairs")
                for i, match in enumerate(faiss_matches[:20]):
                    color = "#ff4b4b" if match["similarity"] >= 0.90 else "#ffa500"
                    with st.expander(
                        f"#{i+1} · {match['source_doc']}  ↔  {match['match_doc']} "
                        f"— {match['similarity']*100:.1f}%", expanded=(i == 0)
                    ):
                        ca, cb = st.columns(2)
                        with ca:
                            st.markdown(f"**📄 {match['source_doc']}**")
                            st.info(match["source_chunk_text"])
                        with cb:
                            st.markdown(f"**📄 {match['match_doc']}**")
                            st.warning(match["match_chunk_text"])
                        st.markdown(
                            f"<div style='text-align:right;'>"
                            f"<span style='background:{color};color:white;padding:3px 12px;"
                            f"border-radius:10px;font-size:0.85rem;font-weight:700;'>"
                            f"Similarity: {match['similarity']*100:.1f}%</span></div>",
                            unsafe_allow_html=True,
                        )
                if len(faiss_matches) > 20:
                    st.caption(f"Showing top 20 of {len(faiss_matches)} matches.")

        st.divider()
        st.subheader("🔎 Query: Search Custom Text Against All Assignments")
        st.caption("Paste any text snippet — FAISS finds the most similar paragraphs across all uploads.")

        query_text = st.text_area("Paste a text snippet:", height=120,
                                  placeholder="Paste a paragraph from a suspected plagiarised source…")
        if st.button("🔍 Search Assignments", key="custom_query") and query_text.strip():
            from src.core.embedding_model import embed_chunks
            with st.spinner("Embedding query and searching…"):
                query_vec = embed_chunks([query_text.strip()])[0]
                results   = search_similar_chunks(query_vec, faiss_index, registry,
                                                  top_k=faiss_top_k, threshold=faiss_threshold)
            
            if selected_class != "All Classes":
                results = [(record, score) for record, score in results if record.doc_name in class_docs]

            if not results:
                st.info("No sufficiently similar chunks found.")
            else:
                st.success(f"Top {len(results)} matches:")
                for rank, (record, score) in enumerate(results, 1):
                    with st.expander(
                        f"#{rank} — {record.doc_name} (chunk #{record.chunk_index+1}) · {score:.1%}",
                        expanded=(rank == 1)
                    ):
                        cq, cm = st.columns(2)
                        with cq:
                            st.markdown("**Your query:**")
                            st.info(query_text.strip())
                        with cm:
                            st.markdown(f"**Match in {record.doc_name}:**")
                            st.warning(record.chunk_text)

    # ══ TAB 3 ════════════════════════════════════════════════════════════════════
    with tab_matrix:
        st.subheader("📋 Similarity Matrix")
        if active_sim_df is None:
            st.info("No similarity matrix available. Please ensure at least 2 documents are uploaded for the selected class.")
        else:
            def _highlight(val: Any) -> str:
                numeric_val = float(val)
                if numeric_val >= 0.90:
                    return "background-color:#ff4b4b;color:white;font-weight:bold;"
                elif numeric_val >= threshold:
                    return "background-color:#ffa500;color:white;font-weight:bold;"
                return ""
            styled_df = active_sim_df.style.format("{:.4f}").map(_highlight)
            st.dataframe(styled_df, use_container_width=True)
            st.download_button("⬇️ Download CSV", active_sim_df.to_csv().encode("utf-8"),
                               "similarity_matrix.csv", "text/csv")

    # ══ TAB 4 ════════════════════════════════════════════════════════════════════
    with tab_heatmap:
     st.subheader("🗺️ Similarity Heatmap")
     if active_sim_df is None:
         st.info("No similarity heatmap or network available. Please ensure at least 2 documents are uploaded for the selected class.")
     else:
         heatmap_fig = plot_similarity_heatmap(
            active_sim_df,
            title="Document Semantic Similarity",
            threshold=threshold,
        )

         st.pyplot(
            heatmap_fig,
            use_container_width=True,
        )

         buf = _io.BytesIO()
         heatmap_fig.savefig(
            buf,
            format="png",
            dpi=150,
            bbox_inches="tight",
        )
         buf.seek(0)

         st.download_button(
           "⬇️ Download Heatmap PNG",
            buf,
            "heatmap.png",
            "image/png",
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

         st.plotly_chart(
            network_fig,
            use_container_width=True,
            config={
                "displaylogo": False,
                "scrollZoom": True,
            },
        )

    # ══ TAB 5 ════════════════════════════════════════════════════════════════════
    with tab_drill:
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
        if n_docs < 2:
            st.warning("Need at least 2 documents.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                doc_a = st.selectbox("Document A", doc_names, index=0, key="da")
            with c2:
                doc_b = st.selectbox(
                    "Document B",
                    [d for d in doc_names if d != doc_a],
                    index=0,
                    key="db"
                )

            score = float(active_sim_df.loc[doc_a, doc_b])
            score_color = "#ff4b4b" if score >= 0.9 else ("#ffa500" if score >= threshold else "#21c55d")
            st.markdown(
                f"**Overall Similarity:** "
                f"<span style='color:{score_color};font-size:1.2rem;font-weight:700;'>"
                f"{score:.1%}</span>", unsafe_allow_html=True,
            )
            st.progress(float(score))
            st.divider()

            emb_a, emb_b     = embeddings.get(doc_a, np.array([])), embeddings.get(doc_b, np.array([]))
            chunks_a, chunks_b = chunked_docs.get(doc_a, []), chunked_docs.get(doc_b, [])

            if emb_a.size > 0 and emb_b.size > 0:
                max_d  = 15
                fig2   = plot_chunk_similarity_comparison(
                    doc_a, doc_b, chunks_a[:max_d], chunks_b[:max_d],
                    cosine_similarity(emb_a, emb_b)[:max_d, :max_d],
                )
                st.pyplot(fig2, use_container_width=True)

                top_pairs = find_most_similar_chunks(
                    chunks_a, chunks_b, emb_a, emb_b, top_k=5, threshold=threshold)
                if top_pairs:
                    st.subheader("🔑 Top Suspicious Paragraph Pairs")
                    for rank, (ca, cb, sim) in enumerate(top_pairs, 1):
                        with st.expander(
                            f"#{rank} — {doc_a} ↔ {doc_b} — {sim:.1%}",
                            expanded=st.session_state.expand_all_drill or (rank == 1),
                        ):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown(f"**From {doc_a}**")
                                st.info(ca)
                            with col2:
                                st.markdown(f"**From {doc_b}**")
                                st.warning(cb)
                    
                    st.divider()
                    st.subheader("📄 Generate PDF Report")
                    st.caption("Download a formal plagiarism report for this document pair.")
                    
                    if st.button("📥 Generate PDF Report", type="primary", use_container_width=True, key="pdf_report"):
                        with st.spinner("Generating PDF report..."):
                            try:
                                _logo_bytes = None
                                _logo_path = st.session_state.get("logo_path")
                                if _logo_path and os.path.exists(_logo_path):
                                    with open(_logo_path, "rb") as _lf:
                                        _logo_bytes = _lf.read()
                                pdf_buffer = generate_plagiarism_report(
                                    doc_a=doc_a,
                                    doc_b=doc_b,
                                    overall_similarity=score,
                                    threshold=threshold,
                                    top_pairs=top_pairs,
                                    report_title="Plagiarism Detection Report",
                                    logo_image=_logo_bytes,
                                    brand_color=st.session_state.get("brand_color"),
                                )
                                st.download_button(
                                    label="⬇️ Download PDF Report",
                                    data=pdf_buffer,
                                    file_name=f"plagiarism_report_{doc_a}_{doc_b}.pdf",
                                    mime="application/pdf",
                                    use_container_width=True
                                )
                            except Exception as e:
                                st.error(f"Error generating PDF report: {str(e)}")
                else:
                    st.success("No paragraph pairs above the threshold for this pair.")

            with st.expander(
                "📄 View Raw Extracted Text",
                expanded=st.session_state.expand_all_drill,
            ):
                t1, t2 = st.columns(2)
                with t1:
                    st.markdown(f"**{doc_a}**")
                    st.text_area("", raw_texts.get(doc_a, "(empty)"), height=300, key="ta")
                with t2:
                    st.markdown(f"**{doc_b}**")
                    st.text_area("", raw_texts.get(doc_b, "(empty)"), height=300, key="tb")

    # ══ TAB 6: Manage Corpus ════════════════════════════════════════════════════
    with tab_corpus:
        st.subheader("🗃️ Corpus Management")
        st.caption("View and delete indexed documents. Deleting a document removes it from the database and rebuilds the FAISS index.")
        
        # Fetch all documents from database
        all_docs = get_all_documents()
        
        if not all_docs:
            st.info("No documents in the corpus database.")
        else:
            # Prepare data for display with chunk counts
            from src.db.corpus_db import get_document_chunks_count
            
            doc_data = []
            for doc in all_docs:
                chunk_count = get_document_chunks_count(doc['filename'])
                doc_data.append({
                    "Filename": doc['filename'],
                    "File Hash": doc['file_hash'][:16] + "...",  # Show truncated hash
                    "Upload Date": doc['upload_date'],
                    "Class Section": doc['class_section'] or "N/A",
                    "Student Name": doc['student_name'] or "N/A",
                    "Assignment": doc['assignment_title'] or "N/A",
                    "Chunk Count": chunk_count
                })
            
            # Display as dataframe
            docs_df = pd.DataFrame(doc_data)
            st.dataframe(docs_df, use_container_width=True, hide_index=True)
            
            st.divider()
            st.subheader("🗑️ Delete Documents")
            st.caption("Select documents to delete from the corpus. This action cannot be undone.")
            
            # Delete document form
            with st.form("delete_document_form"):
                doc_to_delete = st.selectbox(
                    "Select document to delete",
                    options=[doc['filename'] for doc in all_docs],
                    key="corpus_delete_select"
                )
                
                delete_submitted = st.form_submit_button("Delete Document", type="secondary")
                
                if delete_submitted:
                    with st.spinner(f"Deleting {doc_to_delete} and rebuilding FAISS index..."):
                        try:
                            # Delete document from database (cascades to chunks)
                            delete_document(doc_to_delete)
                            
                            # Rebuild FAISS index from remaining embeddings
                            embeddings_matrix = get_all_embeddings()
                            if embeddings_matrix.size > 0:
                                new_index = build_index_from_matrix(embeddings_matrix)
                                save_index(new_index, _INDEX_PATH)
                                st.success(f"✅ Document '{doc_to_delete}' deleted. FAISS index rebuilt with {new_index.ntotal} vectors.")
                            else:
                                # No embeddings left, remove the index file
                                if os.path.exists(_INDEX_PATH):
                                    os.remove(_INDEX_PATH)
                                st.success(f"✅ Document '{doc_to_delete}' deleted. No documents remaining, FAISS index removed.")
                            
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error deleting document: {str(e)}")

    # ══ TAB 6: User Management ═══════════════════════════════════════════════════
    with tab_users:
        st.subheader("👥 User Management")
        st.caption("Manage user accounts and roles. Only administrators can access this panel.")
        
        current_username = st.session_state.get("username", "")
        
        # Display existing users
        users = get_all_users()
        if users:
            users_df = pd.DataFrame(users)
            st.dataframe(users_df, use_container_width=True, hide_index=True)
        else:
            st.info("No users found in database.")
        
        st.divider()
        
        # Add new user form
        st.subheader("Add New User")
        with st.form("add_user_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                new_username = st.text_input("Username")
            with col2:
                new_password = st.text_input("Password", type="password")
            with col3:
                new_role = st.selectbox("Role", ["teacher", "student", "admin"])
            
            add_user_submitted = st.form_submit_button("Add User", type="primary")
            
            if add_user_submitted:
                if new_username and new_password:
                    try:
                        add_user(new_username, new_password, new_role)
                        st.success(f"User '{new_username}' added successfully with role '{new_role}'.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error adding user: {str(e)}")
                else:
                    st.error("Username and password are required.")
        
        st.divider()
        
        # Delete user form
        st.subheader("Delete User")
        with st.form("delete_user_form"):
            if users:
                user_to_delete = st.selectbox("Select user to delete", [u["username"] for u in users if u["username"] != current_username])
                delete_user_submitted = st.form_submit_button("Delete User", type="secondary")
                
                if delete_user_submitted:
                    if user_to_delete == current_username:
                        st.error("You cannot delete your own account.")
                    else:
                        try:
                            delete_user(user_to_delete)
                            st.success(f"User '{user_to_delete}' deleted successfully.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error deleting user: {str(e)}")
            else:
                st.info("No users available to delete.")
        
        st.divider()
        
        # Reset password form
        st.subheader("Reset Password")
        with st.form("reset_password_form"):
            if users:
                user_to_reset = st.selectbox("Select user", [u["username"] for u in users])
                new_password_reset = st.text_input("New Password", type="password")
                reset_password_submitted = st.form_submit_button("Reset Password", type="secondary")
                
                if reset_password_submitted:
                    if new_password_reset:
                        try:
                            update_password(user_to_reset, new_password_reset)
                            st.success(f"Password for '{user_to_reset}' reset successfully.")
                        except Exception as e:
                            st.error(f"Error resetting password: {str(e)}")
                    else:
                        st.error("New password is required.")
            else:
                st.info("No users available.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("🎓 Semantic Plagiarism Detection System · Sentence Transformers + FAISS · Streamlit")