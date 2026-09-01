"""
Document Fingerprinting & Deduplication — Streamlit Dashboard Component
=======================================================================
Interactive dashboard tab for viewing document deduplication results,
cluster analysis, and fingerprint management.
"""

from __future__ import annotations

import io
import json
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional

import streamlit as st

from src.security.document_fingerprint import (
    DedupReport,
    DocumentFingerprint,
    DocumentFingerprintEngine,
    DuplicateCluster,
    DuplicateMatch,
    FingerprintMethod,
    FingerprintStore,
    MatchType,
)
from src.visualization.dedup_charts import (
    create_cluster_size_chart,
    create_duplicate_summary_gauge,
    create_match_heatmap,
    create_match_type_breakdown,
    create_scanning_metrics_table,
    create_similarity_histogram,
    render_dedup_summary_metrics,
)

logger = logging.getLogger(__name__)

KEY_DEDUP_ENGINE = "dedup_fingerprint_engine"
KEY_DEDUP_REPORT = "dedup_last_report"
KEY_DEDUP_LOADED = "dedup_data_loaded"


def initialize_dedup_engine(storage_path: Optional[str] = None) -> DocumentFingerprintEngine:
    """Initialize or retrieve the fingerprint engine from session state."""
    if KEY_DEDUP_ENGINE not in st.session_state:
        st.session_state[KEY_DEDUP_ENGINE] = DocumentFingerprintEngine(
            storage_path=storage_path,
        )
    return st.session_state[KEY_DEDUP_ENGINE]


def _seed_demo_fingerprints(engine: DocumentFingerprintEngine) -> None:
    """Seed the engine with demo documents for development and preview."""
    import random

    random.seed(99)

    # Original documents
    originals = {
        "essay_ai_ethics": (
            "Artificial intelligence raises significant ethical questions in modern society. "
            "The rapid advancement of machine learning algorithms has outpaced regulatory "
            "frameworks designed to protect individual privacy and civil liberties. "
            "As AI systems become more capable of making autonomous decisions, questions "
            "of accountability and transparency become paramount. Researchers must consider "
            "the societal impact of their work and implement safeguards against misuse."
        ),
        "research_ml_pipeline": (
            "Building robust machine learning pipelines requires careful attention to data "
            "quality, feature engineering, and model validation. Cross-validation techniques "
            "help ensure generalization to unseen data. Hyperparameter tuning using grid "
            "search or Bayesian optimization can significantly improve model performance. "
            "Production deployment requires monitoring for data drift and model degradation."
        ),
        "report_cloud_computing": (
            "Cloud computing has transformed how organizations deploy and scale applications. "
            "Serverless architectures reduce operational overhead by eliminating server "
            "management. Container orchestration with Kubernetes enables portable deployments "
            "across cloud providers. Cost optimization requires careful monitoring of "
            "resource utilization and automatic scaling policies."
        ),
        "thesis_nlp_review": (
            "Natural language processing has seen remarkable advances with transformer "
            "architectures. Attention mechanisms allow models to capture long-range "
            "dependencies in text. Pre-trained language models like BERT and GPT have "
            "achieved state-of-the-art results across numerous benchmarks. Transfer "
            "learning enables fine-tuning for domain-specific tasks with limited data."
        ),
        "assignment_data_structures": (
            "Binary search trees provide efficient lookup operations with logarithmic "
            "time complexity. Self-balancing variants like AVL trees and red-black trees "
            "guarantee worst-case performance. Hash tables offer constant-time average "
            "operations but may degrade with poor hash functions. Graph algorithms such "
            "as Dijkstra shortest path are fundamental to network routing."
        ),
    }

    # Create exact and near duplicates
    modified_docs = {}

    # Exact duplicate of essay_ai_ethics
    modified_docs["essay_ai_ethics_copy"] = originals["essay_ai_ethics"]

    # Near duplicate with paraphrasing
    modified_docs["essay_ai_ethics_paraphrased"] = (
        "The advancement of artificial intelligence poses critical ethical challenges "
        "in today's world. The swift progress of machine learning techniques has "
        "surpassed regulatory systems meant to safeguard personal privacy and "
        "democratic freedoms. As AI becomes more autonomous in decision-making, "
        "issues of responsibility and openness become essential. Scientists should "
        "evaluate the societal consequences of their research and establish "
        "protective measures against harmful applications."
    )

    # Near duplicate of research_ml_pipeline
    modified_docs["research_ml_variant"] = (
        "Developing reliable machine learning workflows demands meticulous attention "
        "to data preprocessing, feature selection, and model evaluation. K-fold "
        "cross-validation ensures models generalize well to new data. Optimizing "
        "hyperparameters through grid search or Bayesian methods can substantially "
        "boost model accuracy. Deploying to production requires tracking data "
        "distribution shifts and model performance degradation over time."
    )

    # Near duplicate of report_cloud_computing
    modified_docs["report_cloud_variant"] = (
        "Cloud infrastructure has revolutionized how businesses deploy and scale "
        "software systems. Serverless computing eliminates the need for direct "
        "server management reducing operational costs. Container-based orchestration "
        "through Kubernetes enables consistent deployments across multiple cloud "
        "platforms. Managing costs effectively involves continuous monitoring of "
        "resource usage and implementing auto-scaling strategies."
    )

    # Similar to thesis_nlp_review
    modified_docs["thesis_nlp_variant"] = (
        "Language understanding models have achieved breakthrough results using "
        "transformer-based architectures. Self-attention layers enable the model "
        "to understand relationships between distant tokens in a sequence. "
        "Pre-trained transformers such as BERT and GPT have set new benchmarks "
        "on a wide range of language tasks. Fine-tuning pre-trained models on "
        "small domain datasets has proven highly effective."
    )

    all_docs = {**originals, **modified_docs}

    for doc_id, text in all_docs.items():
        engine.compute_fingerprint(
            text=text,
            document_id=doc_id,
            metadata={"source": "demo", "doc_type": "text"},
        )


def render_dedup_tab(
    documents: Optional[List[Dict[str, str]]] = None,
) -> None:
    """Render the complete Deduplication dashboard tab.

    Args:
        documents: Optional list of dicts with 'id' and 'text' keys.
    """
    engine = initialize_dedup_engine()

    # Load provided documents
    if documents and not st.session_state.get(KEY_DEDUP_LOADED, False):
        for doc in documents:
            engine.compute_fingerprint(
                text=doc.get("text", ""),
                document_id=doc.get("id", f"doc_{engine.store.size}"),
            )
        st.session_state[KEY_DEDUP_LOADED] = True
        if documents:
            st.success(f"Fingerprinted {len(documents)} documents.")

    # Seed demo data if empty
    if engine.store.size == 0:
        _seed_demo_fingerprints(engine)
        st.info(f"Demo data loaded: {engine.store.size} documents fingerprinted.")

    # ── Sidebar Controls ───────────────────────────────────────────────
    with st.sidebar:
        st.markdown("---")
        st.subheader("🔍 Dedup Settings")

        minhash_threshold = st.slider(
            "MinHash Threshold", 0.5, 1.0, 0.85, 0.05,
            key="dedup_mh_thresh",
        )
        engine.minhash_threshold = minhash_threshold

        simhash_hamming = st.slider(
            "SimHash Hamming Distance", 1, 10, 3, 1,
            key="dedup_hamming_thresh",
        )
        engine.simhash_hamming_threshold = simhash_hamming

        trigram_threshold = st.slider(
            "Trigram Threshold", 0.5, 1.0, 0.90, 0.05,
            key="dedup_tri_thresh",
        )
        engine.trigram_threshold = trigram_threshold

        if st.button("🔄 Reset & Reload Demo", key="dedup_reset"):
            engine.store.clear()
            _seed_demo_fingerprints(engine)
            st.session_state.pop(KEY_DEDUP_REPORT, None)
            st.rerun()

    # ── Main Content ───────────────────────────────────────────────────
    st.title("🔎 Document Fingerprinting & Deduplication")
    st.caption(
        f"**{engine.store.size}** documents fingerprinted | "
        f"Thresholds: MH={minhash_threshold:.2f}, SH≤{simhash_hamming}, "
        f"TG={trigram_threshold:.2f}"
    )

    # Run scan
    with st.spinner("Scanning corpus for duplicates..."):
        report = engine.scan_corpus()
    st.session_state[KEY_DEDUP_REPORT] = report

    # ── KPI Metrics ────────────────────────────────────────────────────
    metrics = render_dedup_summary_metrics(report)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Documents", metrics["total_documents"])
    with c2:
        st.metric("Unique", f"{metrics['unique_documents']} ({metrics['unique_pct']}%)")
    with c3:
        st.metric("Exact Duplicates", metrics["exact_duplicates"])
    with c4:
        st.metric("Near Duplicates", metrics["near_duplicates"])

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        st.metric("Clusters", metrics["clusters"])
    with c6:
        st.metric("Avg Cluster Size", metrics["avg_cluster_size"])
    with c7:
        st.metric("Duplicate %", f"{metrics['duplicate_pct']}%")
    with c8:
        st.metric("Scan Time", f"{metrics['scan_time_ms']:.1f} ms")

    st.divider()

    # ── Summary Visualizations ─────────────────────────────────────────
    col_left, col_right = st.columns([1, 1])

    with col_left:
        fig_gauge = create_duplicate_summary_gauge(report)
        if fig_gauge:
            st.plotly_chart(fig_gauge, use_container_width=True, key="dedup_gauge")

    with col_right:
        fig_breakdown = create_match_type_breakdown(report)
        if fig_breakdown:
            st.plotly_chart(fig_breakdown, use_container_width=True, key="dedup_breakdown")

    st.divider()

    # ── Score Distribution & Cluster Sizes ─────────────────────────────
    col_hist, col_clusters = st.columns([3, 2])

    with col_hist:
        fig_hist = create_similarity_histogram(report)
        if fig_hist:
            st.plotly_chart(fig_hist, use_container_width=True, key="dedup_hist")

    with col_clusters:
        fig_clusters = create_cluster_size_chart(report)
        if fig_clusters:
            st.plotly_chart(fig_clusters, use_container_width=True, key="dedup_clusters")

    st.divider()

    # ── Heatmap ────────────────────────────────────────────────────────
    st.subheader("🗺️ Document Similarity Heatmap")
    fig_heatmap = create_match_heatmap(report)
    if fig_heatmap:
        st.plotly_chart(fig_heatmap, use_container_width=True, key="dedup_heatmap")
    else:
        st.info("Not enough matches for heatmap visualization.")

    st.divider()

    # ── Cluster Details ────────────────────────────────────────────────
    st.subheader("📋 Duplicate Clusters")

    if report.clusters:
        for cluster in report.clusters:
            with st.expander(
                f"Cluster {cluster.cluster_id} — {cluster.cluster_size} docs "
                f"({cluster.match_type.value})",
                expanded=False,
            ):
                st.markdown(f"**Representative:** `{cluster.representative_id}`")
                st.markdown(f"**Internal Similarity:** {cluster.internal_similarity:.3f}")
                st.markdown("**Documents:**")
                for doc_id in cluster.document_ids:
                    marker = " ⭐" if doc_id == cluster.representative_id else ""
                    st.markdown(f"  - `{doc_id}`{marker}")
    else:
        st.info("No duplicate clusters detected.")

    st.divider()

    # ── Manual Check ───────────────────────────────────────────────────
    st.subheader("🧪 Manual Duplicate Check")
    st.caption("Enter text to check against the fingerprinted corpus.")

    check_text = st.text_area(
        "Text to check",
        height=120,
        key="dedup_check_text",
        placeholder="Paste document text here...",
    )

    if st.button("🔍 Check for Duplicates", key="dedup_check_btn") and check_text.strip():
        with st.spinner("Fingerprinting and comparing..."):
            matches = engine.find_duplicates(check_text, document_id="manual_query")

        if matches:
            st.warning(f"Found {len(matches)} potential duplicates:")
            for i, match in enumerate(matches[:10], 1):
                st.markdown(
                    f"**{i}.** `{match.target_id}` — "
                    f"Score: **{match.overall_score:.3f}** ({match.match_type.value})  \n"
                    f"    MinHash: {match.minhash_similarity:.3f} | "
                    f"SimHash dist: {match.simhash_hamming_distance} | "
                    f"Trigram: {match.trigram_overlap:.3f}"
                )
        else:
            st.success("No duplicates found — document appears unique.")

    st.divider()

    # ── Add Documents ──────────────────────────────────────────────────
    st.subheader("📥 Add Documents to Corpus")

    add_col1, add_col2 = st.columns([2, 1])
    with add_col1:
        new_doc_id = st.text_input("Document ID", key="dedup_new_id", placeholder="my_document_01")
        new_doc_text = st.text_area(
            "Document Text", height=100, key="dedup_new_text",
            placeholder="Enter or paste document text...",
        )
    with add_col2:
        st.markdown("###")
        if st.button("➕ Add & Fingerprint", key="dedup_add_btn"):
            if new_doc_id.strip() and new_doc_text.strip():
                fp = engine.compute_fingerprint(
                    text=new_doc_text,
                    document_id=new_doc_id.strip(),
                    metadata={"source": "manual_upload"},
                )
                st.success(f"Fingerprinted `{new_doc_id}` (SHA: {fp.sha256_hash[:12]}...)")
                st.rerun()
            else:
                st.error("Please provide both a Document ID and text content.")

    st.divider()

    # ── Export ──────────────────────────────────────────────────────────
    st.subheader("📥 Export Results")

    exp_cols = st.columns(3)
    with exp_cols[0]:
        json_data = engine.export_report_json(report)
        st.download_button(
            "📄 Download JSON Report",
            data=json_data,
            file_name="dedup_report.json",
            mime="application/json",
            key="dedup_export_json",
        )

    with exp_cols[1]:
        summary = json.dumps({
            "total_documents": report.total_documents,
            "unique_documents": report.unique_documents,
            "exact_duplicates": report.exact_duplicate_count,
            "near_duplicates": report.near_duplicate_count,
            "clusters": len(report.clusters),
            "scan_time_ms": report.scan_duration_ms,
        }, indent=2)
        st.download_button(
            "⚡ Download Summary",
            data=summary,
            file_name="dedup_summary.json",
            mime="application/json",
            key="dedup_export_summary",
        )

    with exp_cols[2]:
        # Export fingerprint store
        store_json = json.dumps({
            doc_id: {
                "sha256": fp.sha256_hash,
                "word_count": fp.word_count,
                "char_count": fp.char_count,
            }
            for doc_id, fp in engine.store.get_all().items()
        }, indent=2)
        st.download_button(
            "🗂️ Download Fingerprints",
            data=store_json,
            file_name="fingerprints.json",
            mime="application/json",
            key="dedup_export_fps",
        )
