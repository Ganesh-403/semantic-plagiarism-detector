"""
Cross-Lingual Plagiarism Detection UI Component.

Provides Streamlit-based interface components, language badges,
and dashboard reporting for multi-language plagiarism detection.
"""

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.core.cross_lingual_detector import (
    CrossLingualDetector,
    CrossLingualConfig,
    LANGUAGE_NAMES,
    detect_cross_lingual_plagiarism,
    get_language_name,
)


# ============================================================================
# MAIN DASHBOARD RENDERER
# ============================================================================


def render_cross_lingual_dashboard():
    """Render the main cross-lingual detection dashboard."""
    st.title("🌍 Cross-Lingual Plagiarism Detection")
    st.markdown(
        "Detect plagiarism across documents written in **different languages** using multilingual embeddings."
    )

    tab_config, tab_detect, tab_results = st.tabs(
        ["⚙️ Configuration", "🔍 Detection", "📊 Results"]
    )

    with tab_config:
        _render_configuration()

    with tab_detect:
        _render_detection()

    with tab_results:
        _render_results()


def _render_configuration():
    """Render configuration panel."""
    st.subheader("Cross-Lingual Configuration")

    with st.form("cross_lingual_config"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Language Settings**")
            languages = st.multiselect(
                "Enabled Languages",
                options=[
                    "en",
                    "es",
                    "fr",
                    "de",
                    "pt",
                    "it",
                    "nl",
                    "ru",
                    "zh",
                    "ja",
                    "ar",
                    "hi",
                    "ko",
                    "tr",
                    "pl",
                ],
                default=["en", "es", "fr", "de"],
                format_func=lambda x: (
                    f"{LANGUAGE_NAMES.get(x, get_language_name(x))} ({x})"
                ),
            )
            use_translation = st.checkbox(
                "Use Translation Bridge",
                value=True,
                help="Translate documents to a common language for comparison",
            )

        with col2:
            st.markdown("**Detection Settings**")
            threshold = st.slider("Similarity Threshold", 0.50, 0.99, 0.65, 0.01)
            top_k = st.number_input("Top K Results", 1, 50, 10)
            embedding_model = st.selectbox(
                "Embedding Model",
                [
                    "paraphrase-multilingual-MiniLM-L12-v2",
                    "multilingual-e5-base",
                    "lang-distiluse-base",
                ],
                index=0,
            )

        if st.form_submit_button("💾 Save Configuration", use_container_width=True):
            st.session_state.cross_lingual_config = CrossLingualConfig(
                enabled_languages=languages,
                similarity_threshold=threshold,
                use_translation_bridge=use_translation,
                embedding_model=embedding_model,
                top_k=top_k,
            )
            st.success("✅ Configuration saved!")


def _render_detection():
    """Render detection interface."""
    st.subheader("Cross-Lingual Document Analysis")

    uploaded_files = st.file_uploader(
        "Upload documents (PDF, TXT)",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Upload documents in different languages for cross-lingual comparison",
    )

    if not uploaded_files:
        st.info("👆 Upload at least 2 documents to begin cross-lingual analysis.")
        st.markdown("""
        **How it works:**
        1. 📄 Upload documents in different languages
        2. 🔍 System detects language for each document
        3. 🧠 Multilingual embeddings are generated
        4. 🌐 Cross-language similarity is computed
        5. 📊 Results show matches across languages
        """)
        return

    if len(uploaded_files) < 2:
        st.warning("Please upload at least 2 documents.")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.info(f"📄 **{len(uploaded_files)} documents** uploaded")
    with col2:
        manual_lang = st.checkbox("Manually set languages", value=False)

    if st.button(
        "🚀 Run Cross-Lingual Detection", type="primary", use_container_width=True
    ):
        _run_detection(uploaded_files, manual_lang)


def _run_detection(uploaded_files, manual_lang: bool):
    """Execute cross-lingual detection."""
    config = st.session_state.get("cross_lingual_config", CrossLingualConfig())
    detector = CrossLingualDetector(config)

    documents = {}
    for f in uploaded_files:
        text = f.read().decode("utf-8", errors="ignore")
        if manual_lang:
            lang = st.selectbox(
                f"Language for {f.name}",
                list(LANGUAGE_NAMES.keys()),
                format_func=lambda x: LANGUAGE_NAMES.get(x, get_language_name(x)),
                key=f"lang_{f.name}",
            )
        else:
            lang = detector.detect_language(text)

        chunks = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 50]
        documents[f.name] = (lang, chunks)

    with st.spinner("🌍 Analyzing cross-lingual plagiarism..."):
        result = detector.detect_cross_lingual_plagiarism(documents)

    st.session_state.cross_lingual_result = result
    st.success(
        f"✅ Analysis complete! Found **{len(result.matches)}** matches across **{len(result.language_distribution)}** languages."
    )


def _render_results():
    """Render detection results."""
    st.subheader("Cross-Lingual Detection Results")

    result = st.session_state.get("cross_lingual_result")
    if not result:
        st.info("Run a detection first to see results here.")
        return

    data = result.to_dict()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Documents", data["summary"]["total_documents"])
    col2.metric("Languages", data["summary"]["languages_detected"])
    col3.metric("Cross-Lingual Matches", data["summary"]["cross_lingual_matches"])
    col4.metric("High Severity", data["summary"]["high_severity"])

    st.subheader("🌐 Language Distribution")
    lang_data = [
        {"Language": LANGUAGE_NAMES.get(k, get_language_name(k)), "Count": v}
        for k, v in data["language_distribution"].items()
    ]
    if lang_data:
        fig_lang = px.pie(
            lang_data, values="Count", names="Language", title="Documents by Language"
        )
        st.plotly_chart(fig_lang, use_container_width=True)

    st.subheader("🔍 Detected Matches")
    if data["matches"]:
        matches_df = pd.DataFrame(data["matches"])
        display_cols = [
            "source_doc",
            "source_lang",
            "target_doc",
            "target_lang",
            "similarity",
            "method",
            "translation_used",
        ]
        if all(c in matches_df.columns for c in display_cols):
            display_df = matches_df[display_cols].copy()
            display_df.columns = [
                "Source Doc",
                "Source Lang",
                "Target Doc",
                "Target Lang",
                "Similarity",
                "Method",
                "Translated",
            ]
            display_df["Similarity"] = display_df["Similarity"].apply(
                lambda x: f"{x:.1%}"
            )

            st.dataframe(display_df, use_container_width=True)

        if data["matches"]:
            st.subheader("📊 Similarity Distribution")
            sim_values = [m["similarity"] for m in data["matches"]]
            fig_sim = px.histogram(
                x=sim_values,
                nbins=20,
                title="Match Similarity Distribution",
                labels={"x": "Similarity Score", "y": "Count"},
            )
            fig_sim.update_traces(marker_color="#60a5fa")
            st.plotly_chart(fig_sim, use_container_width=True)

        st.subheader("📝 Match Details")
        for i, match in enumerate(data["matches"][:10]):
            color = "#ef4444" if match["similarity"] >= 0.9 else "#f97316"
            with st.expander(
                f"#{i + 1} · {match['source_doc']} ({match['source_lang']}) ↔ "
                f"{match['target_doc']} ({match['target_lang']}) — {match['similarity']:.1%}",
                expanded=(i == 0),
            ):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(
                        f"**📄 {match['source_doc']}** ({LANGUAGE_NAMES.get(match['source_lang'], get_language_name(match['source_lang']))})"
                    )
                    st.info(
                        match["source_chunk"][:500]
                        if match["source_chunk"]
                        else "(empty)"
                    )
                with c2:
                    st.markdown(
                        f"**📄 {match['target_doc']}** ({LANGUAGE_NAMES.get(match['target_lang'], get_language_name(match['target_lang']))})"
                    )
                    st.warning(
                        match["target_chunk"][:500]
                        if match["target_chunk"]
                        else "(empty)"
                    )
                st.markdown(
                    f"<div style='text-align:right;'>"
                    f"<span style='background:{color};color:white;padding:3px 12px;"
                    f"border-radius:10px;font-size:0.85rem;font-weight:700;'>"
                    f"Similarity: {match['similarity']:.1%} | "
                    f"Confidence: {match['confidence']:.1%} | "
                    f"{'🌐 Cross-Lingual' if match['translation_used'] else '📝 Same-Language'}"
                    f"</span></div>",
                    unsafe_allow_html=True,
                )
    else:
        st.success("✅ No cross-lingual plagiarism matches found above the threshold.")

    st.subheader("⬇️ Export Results")
    col1, col2 = st.columns(2)
    with col1:
        if data["matches"]:
            csv = pd.DataFrame(data["matches"]).to_csv(index=False)
            st.download_button(
                "Download CSV", csv, "cross_lingual_results.csv", "text/csv"
            )
    with col2:
        if data["matches"]:
            import json

            st.download_button(
                "Download JSON",
                json.dumps(data, indent=2, default=str),
                "cross_lingual_results.json",
                "application/json",
            )


# ============================================================================
# LANGUAGE BADGE RENDERERS & HELPER UI FUNCTIONS
# ============================================================================


def render_language_badge(lang_code: str, show_name: bool = True) -> str:
    """Generate HTML badge for language display."""
    lang_name = get_language_name(lang_code)
    colors = {
        "en": "#3B82F6",
        "es": "#F59E0B",
        "fr": "#10B981",
        "de": "#EF4444",
        "it": "#8B5CF6",
        "pt": "#EC4899",
        "nl": "#06B6D4",
        "ru": "#6B7280",
        "zh": "#EF4444",
        "ja": "#F472B6",
        "ko": "#8B5CF6",
        "ar": "#059669",
        "hi": "#F97316",
        "tr": "#3B82F6",
        "pl": "#F472B6",
    }
    color = colors.get(lang_code, "#6B7280")
    if show_name:
        return f'<span style="background:{color};color:white;padding:2px 10px;border-radius:12px;font-size:0.75rem;font-weight:500;">🌐 {lang_name}</span>'
    return f'<span style="background:{color};color:white;padding:2px 8px;border-radius:12px;font-size:0.7rem;font-weight:500;">{lang_code.upper()}</span>'


def render_translation_indicator(
    is_translated: bool, lang_code: Optional[str] = None
) -> str:
    """Generate HTML indicator for translated content."""
    if not is_translated:
        return ""
    lang_name = get_language_name(lang_code) if lang_code else "Unknown"
    return f'<span style="background:#3B82F6;color:white;padding:2px 10px;border-radius:12px;font-size:0.7rem;font-weight:500;margin-left:8px;">🔄 Translated from {lang_name}</span>'


def render_cross_lingual_settings() -> bool:
    """Render cross-lingual settings toggle in sidebar/panel."""
    enabled = st.toggle(
        "🌐 Enable Cross-Lingual Detection",
        value=st.session_state.get("cross_lingual_mode_toggle", False),
        key="cross_lingual_mode_toggle",
    )
    return enabled


def get_cross_lingual_metadata() -> Dict[str, List[Dict[str, Any]]]:
    """Get cross-lingual metadata from session state."""
    return st.session_state.get("translation_metadata", {})


def is_cross_lingual_enabled() -> bool:
    """Check if cross-lingual mode is enabled."""
    return st.session_state.get("cross_lingual_mode_toggle", False)


__all__ = [
    "render_cross_lingual_dashboard",
    "render_language_badge",
    "render_translation_indicator",
    "render_cross_lingual_settings",
    "get_cross_lingual_metadata",
    "is_cross_lingual_enabled",
]
