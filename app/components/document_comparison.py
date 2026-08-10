"""
document_comparison.py
-----------------------
Streamlit component for side-by-side custom document/text comparison.
Provides a clear confirmation popover to prevent accidental data loss.
"""

import streamlit as st
from src.utils.diff_highlighter import highlight_overlap


def render_document_comparison():
    """Render the side-by-side Document Comparison View with Clear Comparison popover."""
    st.subheader("🔬 Custom Document Comparison")
    st.caption("Paste the contents of two documents below to perform a side-by-side similarity highlighting comparison.")

    # Initialize session state for the document inputs
    if "comp_doc_a" not in st.session_state:
        st.session_state["comp_doc_a"] = ""
    if "comp_doc_b" not in st.session_state:
        st.session_state["comp_doc_b"] = ""

    # Inputs Layout
    col1, col2 = st.columns(2)
    with col1:
        doc_a = st.text_area(
            "Document A Text",
            value=st.session_state["comp_doc_a"],
            placeholder="Paste text of the first document here...",
            key="comp_doc_a_input",
            height=200,
        )
    with col2:
        doc_b = st.text_area(
            "Document B Text",
            value=st.session_state["comp_doc_b"],
            placeholder="Paste text of the second document here...",
            key="comp_doc_b_input",
            height=200,
        )

    # Sync text inputs to session state
    st.session_state["comp_doc_a"] = doc_a
    st.session_state["comp_doc_b"] = doc_b

    # Add confirmation popover st.popover("Clear Comparison")
    with st.popover("Clear Comparison"):
        st.write("⚠️ Are you sure you want to clear staged documents?")
        if st.button("Yes, Clear", key="confirm_clear_comp_btn", use_container_width=True):
            st.session_state["comp_doc_a"] = ""
            st.session_state["comp_doc_b"] = ""
            # Reset values inside the text_area widgets
            if "comp_doc_a_input" in st.session_state:
                st.session_state["comp_doc_a_input"] = ""
            if "comp_doc_b_input" in st.session_state:
                st.session_state["comp_doc_b_input"] = ""
            st.rerun()

    # Results Layout
    if doc_a.strip() and doc_b.strip():
        st.markdown("#### 🔍 Highlighted Overlaps")
        highlighted_a, highlighted_b = highlight_overlap(doc_a, doc_b)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Document A (Extracted Overlap)**")
            st.markdown(
                f"<div style='border: 1px solid #e2e8f0; padding: 12px; border-radius: 8px; min-height: 150px; white-space: pre-wrap; background-color: var(--surface);'>{highlighted_a}</div>",
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown("**Document B (Extracted Overlap)**")
            st.markdown(
                f"<div style='border: 1px solid #e2e8f0; padding: 12px; border-radius: 8px; min-height: 150px; white-space: pre-wrap; background-color: var(--surface);'>{highlighted_b}</div>",
                unsafe_allow_html=True,
            )
    elif doc_a.strip() or doc_b.strip():
        st.info("Please paste content into both inputs to run side-by-side comparison.")
