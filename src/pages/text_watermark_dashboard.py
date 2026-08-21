"""Streamlit Dashboard Page for Adversarial Text Watermark Detector Suite."""

import streamlit as st
from src.components.text_watermark_card import render_watermark_card
from src.components.text_watermark_timeline import render_watermark_timeline
from src.services.text_watermark_engine import AdversarialWatermarkEngine


def render_adversarial_watermark_dashboard():
    """Main rendering function for Streamlit Adversarial Watermark dashboard tab."""
    st.set_page_config(page_title="Adversarial Text Watermark Detector", layout="wide")

    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, #0F172A 0%, #7C2D12 100%);
            padding: 32px;
            border-radius: 24px;
            border: 1px solid #334155;
            margin-bottom: 28px;
        ">
            <span style="
                background: rgba(245, 158, 11, 0.15);
                border: 1px solid rgba(245, 158, 11, 0.4);
                color: #FCD34D;
                font-size: 12px;
                font-weight: 800;
                padding: 4px 14px;
                border-radius: 9999px;
            ">
                Statistical LLM Fingerprint Detection
            </span>
            <h1 style="color: white; font-weight: 900; font-size: 36px; margin-top: 12px; margin-bottom: 8px;">
                Adversarial Text Watermark & LLM Fingerprint Detector
            </h1>
            <p style="color: #94A3B8; font-size: 16px; margin: 0;">
                Detect statistical Kirchenbauer z-score logit watermarks and green-list token entropy bias in AI-generated text.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "watermark_matches" not in st.session_state:
        st.session_state["watermark_matches"] = []

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Text Snippet for Watermark Hypothesis Test")
        doc_title = st.text_input(
            "Document Title", value="LLM Generated Physics Summary"
        )
        text_content = st.text_area(
            "Input Text Content",
            value="Quantum mechanics is a fundamental theory in physics that provides a description of the physical properties of nature at the scale of atoms and subatomic particles. It is the foundation of all quantum physics including quantum chemistry, quantum field theory, quantum technology, and quantum information science.",
            height=180,
        )

    with col2:
        st.subheader("Statistical Hyperparameters")
        gamma = st.slider(
            "Green-List Ratio (gamma)",
            min_value=0.10,
            max_value=0.90,
            value=0.50,
            step=0.05,
        )
        z_thresh = st.slider(
            "z-Score Detection Threshold",
            min_value=1.5,
            max_value=6.0,
            value=4.0,
            step=0.1,
        )

    if st.button("Run Statistical Watermark Detection", use_container_width=True):
        match = AdversarialWatermarkEngine.analyze_text(
            document_id="DOC-WM-101",
            document_title=doc_title,
            text_content=text_content,
            gamma=gamma,
            z_threshold=z_thresh,
        )

        st.session_state["watermark_matches"].append(match)

        if match.is_watermark_present:
            st.error(
                f"Watermark Detected! z-score: {match.z_score} (Confidence: {match.watermark_confidence_percentage}%)"
            )
        else:
            st.success(
                f"No Watermark Detected. z-score: {match.z_score} (Unwatermarked / Human text)"
            )

    if st.session_state["watermark_matches"]:
        st.markdown("### Detection Audit Results")
        for match in st.session_state["watermark_matches"]:
            st.markdown(
                render_watermark_card(
                    match if isinstance(match, dict) else match.__dict__
                ),
                unsafe_allow_html=True,
            )

        st.markdown(
            render_watermark_timeline(st.session_state["watermark_matches"]),
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    render_adversarial_watermark_dashboard()
