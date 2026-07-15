import sys, os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import io as _io
import time
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from typing import Any
from utils.warning_list import render_warning_controls
from src.core.document_parser import extract_text_from_pdf
from src.core.text_chunking import chunk_documents
from src.core.embedding_model import embed_documents
from src.core.similarity import (
    document_similarity_matrix, flag_plagiarism,
    find_most_similar_chunks, PLAGIARISM_THRESHOLD,
)
from src.visualization.heatmap import plot_similarity_heatmap, plot_chunk_similarity_comparison
from src.core.faiss_index import build_index, find_plagiarised_chunks, search_similar_chunks
from src.core.webhook import send_plagiarism_alert

# Must be the first Streamlit command called
st.set_page_config(
    page_title="Semantic Plagiarism Detector",
    page_icon="🔍", layout="wide",
    initial_sidebar_state="expanded",
)
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
        for key in ["authenticated", "role", "last_interaction"]:
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
            # Secure hardcoded credentials for demonstrating roles
            if username == "admin" and password == "admin123":
                st.session_state.authenticated = True
                st.session_state.role = "admin"
                st.session_state.last_interaction = time.time()
                st.success("Welcome back, Administrator!")
                st.rerun()
            elif username == "student" and password == "student123":
                st.session_state.authenticated = True
                st.session_state.role = "user"
                st.session_state.last_interaction = time.time()
                st.success("Successfully logged in as Student.")
                st.rerun()
            else:
                st.error("Invalid Username or Password.")
    st.stop()

# Initialize session page if not exists
if "current_page" not in st.session_state:
    st.session_state.current_page = "main"

# Custom routing interceptor
if st.session_state.current_page == "about":
    from app.about_page import render_about_page
    render_about_page()
    st.stop()

# Get secure role for this active interaction
user_role = st.session_state.get("role", "user")

# ── Sidebar (ROLE RESTRICTED Settings) ────────────────────────────────────────
with st.sidebar:
    st.markdown("<div style='font-size: 72px; line-height: 1;'>🕵️‍♂️</div>", unsafe_allow_html=True)
    st.title("⚙️ Settings")
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
    st.markdown("""
**How it works**
1. PDFs parsed with **PyPDF2**
2. Text split into **paragraph chunks**
3. Chunks embedded with **all-MiniLM-L6-v2**
4. **FAISS index** built over all chunk vectors
5. Pairwise **cosine similarity** computed
6. Pairs above threshold flagged
""")
    st.markdown("---")
    st.caption("Semantic Plagiarism Detector · FAISS edition")
    
    # Log out button
    if st.button("🚪 Log Out", use_container_width=True):
        for key in ["authenticated", "role", "last_interaction"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🔍 Semantic Plagiarism Detection System")
st.markdown(
    "Upload student PDFs. Detects **semantic similarity** (even paraphrased text) "
    "using transformer embeddings + **FAISS vector search**."
)
st.divider()

# ── MAIN APPLICATION SECTIONS (ROLE CHECKED) ──────────────────────────────────

if user_role != "admin":
    # STANDARD USER VIEW: Student Query / Search Panel Only (No admin PDF uploading)
    st.subheader("🔎 Secure Student Search Portal")
    st.caption("Paste a text snippet below to check its similarity against existing indexed assignments.")
    
    st.info("🔒 Note: Direct assignment uploads and detailed breakdown panels are restricted to Administrator access.")
    
    query_text = st.text_area("Paste a text snippet to check against index:", height=150,
                              placeholder="Paste a paragraph here to check for plagiarism...")
    
    if st.button("🔍 Run Quick Verification", key="user_query") and query_text.strip():
        st.warning("To query database resources, please coordinate with your Course Administrator.")
else:
    # ADMINISTRATOR ACCESS: Full Upload Pipeline & Evaluation Dashboards
    uploaded_files = st.file_uploader(
        "📂 Upload Assignment PDFs", type=["pdf"],
        accept_multiple_files=True, help="Upload 2 or more PDF files.",
    )
    if not uploaded_files or len(uploaded_files) < 2:
        st.info("👆 Please upload **at least 2** PDF assignment files to begin.")
        st.stop()

    # ── Pipeline (cached) ─────────────────────────────────────────────────────────
    @st.cache_data(show_spinner=False)
    def run_pipeline(file_bytes_dict: dict):
        raw_texts = {
            name: extract_text_from_pdf(_io.BytesIO(data))
            for name, data in file_bytes_dict.items()
        }
        chunked_docs = chunk_documents(raw_texts)
        embeddings   = embed_documents(chunked_docs)
        sim_df       = document_similarity_matrix(embeddings)

        names = list(embeddings.keys())
        n     = len(names)
        chunk_mat = np.zeros((n, n))
        for i, na in enumerate(names):
            for j, nb in enumerate(names):
                if i == j:
                    chunk_mat[i, j] = 1.0
                elif j > i:
                    ea, eb = embeddings[na], embeddings[nb]
                    score  = float(np.max(cosine_similarity(ea, eb))) if ea.size and eb.size else 0.0
                    chunk_mat[i, j] = chunk_mat[j, i] = score
        chunk_sim_df = pd.DataFrame(chunk_mat, index=names, columns=names)

        faiss_index, registry = build_index(embeddings, chunked_docs)
        return raw_texts, chunked_docs, embeddings, sim_df, chunk_sim_df, faiss_index, registry

    file_bytes_dict = {f.name: f.read() for f in uploaded_files}

    with st.spinner("🧠 Processing PDFs, building embeddings and FAISS index…"):
        raw_texts, chunked_docs, embeddings, sim_df, chunk_sim_df, faiss_index, registry = \
            run_pipeline(file_bytes_dict)

    # Check for empty PDFs (e.g. scanned images with no OCR)
    empty_docs = [name for name, text in raw_texts.items() if not text.strip()]
    if empty_docs:
        st.warning(f"⚠️ **Could not extract text from:** {', '.join(empty_docs)}. These might be scanned images or password-protected PDFs.")

    active_sim_df = chunk_sim_df if use_chunk_matrix else sim_df
    flags         = flag_plagiarism(active_sim_df, threshold=threshold)

    # ── Webhook notifications for high-similarity matches (>= 90%) ───────────────
    if "notified_pairs" not in st.session_state:
        st.session_state.notified_pairs = set()
        
    current_files = sorted(list(file_bytes_dict.keys()))
    if "last_uploaded_files" not in st.session_state or st.session_state.last_uploaded_files != current_files:
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
    total_pairs  = n_docs * (n_docs - 1) // 2
    n_flagged    = len(flags)
    n_high       = sum(1 for f in flags if "High" in f["severity"])
    avg_sim      = active_sim_df.values[np.triu_indices(n_docs, k=1)].mean() if n_docs > 1 else 0.0
    total_chunks = sum(len(v) for v in chunked_docs.values())

    col1.metric("📄 Documents",   n_docs)
    col2.metric("🔗 Pairs",       total_pairs)
    col3.metric("🚨 Flagged",     n_flagged,
                delta=f"{n_high} High" if n_high else None, delta_color="inverse")
    col4.metric("📈 Avg Similarity", f"{avg_sim:.1%}")
    col5.metric("🗂️ FAISS Vectors", faiss_index.ntotal)
    st.divider()

    # ── Tabs ──────────────────────────────────────────────────────────────────────
    tab_warnings, tab_faiss, tab_matrix, tab_heatmap, tab_drill = st.tabs([
        "⚠️ Plagiarism Warnings",
        "⚡ FAISS Chunk Search",
        "📋 Similarity Matrix",
        "🗺️ Heatmap",
        "🔬 Pair Drill-Down",
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
                            st.markdown("**Your query:**"); st.info(query_text.strip())
                        with cm:
                            st.markdown(f"**Match in {record.doc_name}:**"); st.warning(record.chunk_text)

    # ══ TAB 3 ════════════════════════════════════════════════════════════════════
    with tab_matrix:
        st.subheader("📋 Similarity Matrix")
        def _highlight(val: Any) -> str:
            numeric_val = float(val)
            if numeric_val >= 0.90:         return "background-color:#ff4b4b;color:white;font-weight:bold;"
            elif numeric_val >= threshold:  return "background-color:#ffa500;color:white;font-weight:bold;"
            return ""
        styled_df = active_sim_df.style.format("{:.4f}").map(_highlight)
        st.dataframe(styled_df, use_container_width=True)
        st.download_button("⬇️ Download CSV", active_sim_df.to_csv().encode("utf-8"),
                           "similarity_matrix.csv", "text/csv")

    # ══ TAB 4 ════════════════════════════════════════════════════════════════════
    with tab_heatmap:
        st.subheader("🗺️ Similarity Heatmap")
        fig = plot_similarity_heatmap(active_sim_df, title="Document Semantic Similarity",
                                      threshold=threshold)
        st.pyplot(fig, use_container_width=True)
        buf = _io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        st.download_button("⬇️ Download PNG", buf, "heatmap.png", "image/png")

    # ══ TAB 5 ════════════════════════════════════════════════════════════════════
    with tab_drill:
        st.subheader("🔬 Pair Drill-Down")
        st.caption("Inspect chunk-level similarity between any two documents.")
        if n_docs < 2:
            st.warning("Need at least 2 documents.")
        else:
            c1, c2 = st.columns(2)
            with c1: doc_a = st.selectbox("Document A", doc_names, index=0, key="da")
            with c2: doc_b = st.selectbox("Document B",
                                           [d for d in doc_names if d != doc_a], index=0, key="db")

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
                        with st.expander(f"#{rank} — Similarity: {sim:.1%}", expanded=(rank == 1)):
                            col1, col2 = st.columns(2)
                            with col1: st.markdown(f"**From {doc_a}**"); st.info(ca)
                            with col2: st.markdown(f"**From {doc_b}**"); st.warning(cb)
                else:
                    st.success("No paragraph pairs above the threshold for this pair.")

            with st.expander("📄 View Raw Extracted Text"):
                t1, t2 = st.columns(2)
                with t1:
                    st.markdown(f"**{doc_a}**")
                    st.text_area("", raw_texts.get(doc_a, "(empty)"), height=300, key="ta")
                with t2:
                    st.markdown(f"**{doc_b}**")
                    st.text_area("", raw_texts.get(doc_b, "(empty)"), height=300, key="tb")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()

# Setup interactive multi-column footer layout
f_col1, f_col2, f_col3, f_col4, f_col5 = st.columns(5)

with f_col1:
    if st.button("🕵️‍♂️ About / Info", key="footer_about", use_container_width=True):
        st.session_state.current_page = "about"
        st.rerun()

with f_col2:
    if st.button("🚧 Help & FAQs", key="footer_help", use_container_width=True):
        st.info("Help page is under construction.")

with f_col3:
    if st.button("🚧 Terms of Use", key="footer_terms", use_container_width=True):
        st.info("Terms & Conditions page is under construction.")

with f_col4:
    if st.button("🚧 Privacy Policy", key="footer_privacy", use_container_width=True):
        st.info("Privacy Policy page is under construction.")

with f_col5:
    if st.button("🚧 Security", key="footer_security", use_container_width=True):
        st.info("Security Policy page is under construction.")

st.markdown("<br>", unsafe_allow_html=True)
st.caption("🎓 Semantic Plagiarism Detection System · Sentence Transformers + FAISS · Streamlit")