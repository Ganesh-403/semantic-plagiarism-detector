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

"""Streamlit Dashboard Page for Multimodal Image Document OCR Plagiarism Suite."""

import streamlit as st

from src.components.multimodal_ocr_card import render_ocr_match_card
from src.components.multimodal_ocr_timeline import render_ocr_timeline
from src.services.multimodal_ocr_engine import MultimodalOcrEngine


def render_multimodal_ocr_dashboard():
    """Main rendering function for Streamlit Multimodal OCR dashboard tab."""
    st.set_page_config(page_title="Multimodal OCR Plagiarism Detector", layout="wide")

    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, #0F172A 0%, #064E3B 100%);
            padding: 32px;
            border-radius: 24px;
            border: 1px solid #334155;
            margin-bottom: 28px;
        ">
            <span style="
                background: rgba(16, 185, 129, 0.15);
                border: 1px solid rgba(16, 185, 129, 0.4);
                color: #6EE7B7;
                font-size: 12px;
                font-weight: 800;
                padding: 4px 14px;
                border-radius: 9999px;
            ">
                Optical Character Recognition & Image Analysis
            </span>
            <h1 style="color: white; font-weight: 900; font-size: 36px; margin-top: 12px; margin-bottom: 8px;">
                Multimodal Image Document OCR Plagiarism Detector
            </h1>
            <p style="color: #94A3B8; font-size: 16px; margin: 0;">
                Extract text from scanned document images, photos, and PDFs using OCR engines to detect cross-modal plagiarism against target text corpora.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "ocr_matches" not in st.session_state:
        st.session_state["ocr_matches"] = []

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Source Document Image OCR Text")
        img_name = st.text_input(
            "Image Document Name", value="Scanned_Assignment_Page_1.jpg"
        )
        raw_ocr = st.text_area(
            "Extracted OCR Text Content",
            value="Artificial intelligence algorithms enable modern automated plagiarism detection tools to parse complex text corpora and identify paraphrased content across multiple languages.",
            height=160,
        )
        engine_type = st.selectbox(
            "OCR Engine Pipeline", ["Tesseract-5.0", "EasyOCR", "PaddleOCR-v3"]
        )

    with col2:
        st.subheader("Reference Text Corpus")
        ref_title = st.text_input(
            "Reference Corpus Title", value="Academic AI Research Paper v2"
        )
        ref_text = st.text_area(
            "Reference Text Content",
            value="Modern artificial intelligence tools use advanced algorithms to parse large document corpora and detect paraphrased text across diverse linguistic sources.",
            height=160,
        )

    if st.button("Run Multimodal OCR Plagiarism Scan", use_container_width=True):
        chunks = MultimodalOcrEngine.process_document_image(
            "IMG-101", img_name, raw_ocr
        )

        match = MultimodalOcrEngine.scan_image_against_reference(
            image_id="IMG-101",
            image_name=img_name,
            raw_ocr_text=raw_ocr,
            reference_id="REF-DOC-88",
            reference_title=ref_title,
            reference_text=ref_text,
            ocr_engine=engine_type,
        )

        st.session_state["ocr_matches"].append(match)

        score_pct = int(match.overall_multimodal_score * 100)
        if score_pct > 70:
            st.error(
                f"High OCR Plagiarism Match Detected! Overall Score: {score_pct}% (Text Sim: {int(match.ocr_text_similarity * 100)}%)"
            )
        else:
            st.success(
                f"Low OCR Match. Overall Score: {score_pct}% (Text Sim: {int(match.ocr_text_similarity * 100)}%)"
            )

    if st.session_state["ocr_matches"]:
        st.markdown("### Scan Results")
        for match in st.session_state["ocr_matches"]:
            st.markdown(
                render_ocr_match_card(
                    match if isinstance(match, dict) else match.__dict__
                ),
                unsafe_allow_html=True,
            )

        st.markdown(
            render_ocr_timeline(st.session_state["ocr_matches"]),
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    render_multimodal_ocr_dashboard()
