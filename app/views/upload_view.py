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
Upload & Document Ingestion View Component.

Renders file upload controls, Google Drive import, document validation,
and Student Quick Verification search portal.
"""

import hashlib
import os

import streamlit as st

from app.session_keys import SessionKeys
from src.core.embedding_model import embed_chunks
from src.core.faiss_index import (
    build_index_from_matrix,
    load_index,
    search_similar_chunks,
)
from src.db.corpus_db import (
    get_all_embeddings,
    get_chunk_registry,
    get_document_by_hash,
)
from src.i18n.translator import get_text
from src.security.metadata_stripper import strip_exif_metadata
from src.utils.filename import (
    InvalidFileExtensionError,
    sanitize_filename,
    unique_filename,
    validate_document_extension,
)
from src.utils.tar_processor import process_tar_file
from src.utils.zip_processor import process_zip_file

try:
    from src.utils.google_drive import bulk_download_drive_folder
except Exception:
    bulk_download_drive_folder = None

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10MB limit

ARCHIVE_EXTENSIONS = (".tar.gz", ".tar.bz2", ".zip")


def render_student_portal(threshold: float, faiss_top_k: int):
    """Render non-admin student quick verification search portal."""
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

                    query_vec = embed_chunks([query_text.strip()])[0]

                    results = search_similar_chunks(
                        query_vec,
                        faiss_index,
                        registry,
                        top_k=faiss_top_k,
                        threshold=threshold,
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
                                doc_id_map[
                                    record.doc_name
                                ] = f"Document-{anon_counter:03d}"
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


def _is_tar_archive(filename: str) -> bool:
    """Return True only for supported compressed TAR archive names."""
    normalized = filename.strip().casefold()
    return normalized.endswith((".tar.gz", ".tar.bz2"))


def _extract_archive_upload(
    original_name: str,
    file_bytes: bytes,
    existing_names: dict,
) -> dict[str, bytes]:
    """Extract a supported archive into uniquely named pipeline documents."""
    if _is_tar_archive(original_name):
        extracted = process_tar_file(file_bytes)
    else:
        extracted = process_zip_file(file_bytes)

    extracted_files: dict[str, bytes] = {}
    for member_name, member_bytes in extracted.items():
        safe_name = unique_filename(member_name, existing_names)
        existing_names[safe_name] = member_bytes
        extracted_files[safe_name] = member_bytes

    return extracted_files


def render_upload_section(user_role: str, lang_code: str, index_path: str):
    """Render document upload section, Google Drive integration, and return file_bytes_dict."""
    if user_role != "admin":
        render_student_portal(
            threshold=st.session_state.get(SessionKeys.THRESHOLD_SLIDER, 0.59),
            faiss_top_k=st.session_state.get(SessionKeys.FAISS_TOP_K_SLIDER, 5),
        )
        return {}

    if os.path.exists(index_path):
        faiss_index = load_index(index_path)
        registry = get_chunk_registry()
        if faiss_index is not None and faiss_index.ntotal != len(registry):
            all_embs = get_all_embeddings()
            if len(all_embs) > 0 and len(all_embs) == len(registry):
                faiss_index = build_index_from_matrix(all_embs)
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

    uploaded_files = st.file_uploader(
        get_text("upload_title", lang=lang_code),
        type=[
            "pdf",
            "docx",
            "txt",
            "md",
            "markdown",
            "mdown",
            "zip",
            "tar.gz",
            "tar.bz2",
            "csv",
        ],
        accept_multiple_files=True,
        key="file_uploader",
    )

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
            is_tar_archive = _is_tar_archive(original_name)
            is_zip_archive = original_name.strip().casefold().endswith(".zip")

            if not (is_tar_archive or is_zip_archive):
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

            file_bytes = uploaded_file.read()

            if is_tar_archive or is_zip_archive:
                try:
                    extracted_files = _extract_archive_upload(
                        original_name,
                        file_bytes,
                        file_bytes_dict,
                    )
                except ValueError as exc:
                    st.error(
                        f"⚠️ Archive **'{sanitize_filename(original_name)}'** was rejected: {exc}"
                    )
                    continue
                except Exception as exc:
                    st.error(
                        f"⚠️ Failed to extract archive **'{sanitize_filename(original_name)}'**: {exc}"
                    )
                    continue

                if not extracted_files:
                    st.warning(
                        f"⚠️ Archive **'{sanitize_filename(original_name)}'** contains no supported documents."
                    )
                    continue

                for member_name, member_bytes in extracted_files.items():
                    file_hash = hashlib.sha256(member_bytes).hexdigest()
                    existing_doc = get_document_by_hash(file_hash)

                    if existing_doc:
                        st.warning(
                            f"⚠️ File **'{member_name}'** from **'{original_name}'** "
                            f"is identical to **'{existing_doc}'** already in the database."
                        )
                        continue

                    file_bytes_dict[member_name] = strip_exif_metadata(
                        member_bytes, member_name
                    )

                st.success(
                    f"📦 Extracted {len(extracted_files)} supported document(s) "
                    f"from **'{original_name}'**."
                )
                continue

            file_hash = hashlib.sha256(file_bytes).hexdigest()
            existing_doc = get_document_by_hash(file_hash)

            if existing_doc:
                st.warning(
                    f"⚠️ File **'{original_name}'** is identical to **'{existing_doc}'** already in the database."
                )
                action = st.radio(
                    f"Action for duplicate file '{original_name}':",
                    ["Skip", "Reprocess"],
                    key=f"dup_{file_hash}_{original_name}",
                    horizontal=True,
                )
                if action == "Skip":
                    continue

            file_bytes_dict[safe_name] = strip_exif_metadata(file_bytes, safe_name)

    for drive_name, drive_bytes in st.session_state.get(
        "drive_imported_files", {}
    ).items():
        safe_drive_name = unique_filename(drive_name, file_bytes_dict)
        file_bytes_dict[safe_drive_name] = drive_bytes

    return file_bytes_dict
