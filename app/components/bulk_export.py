from typing import List

import streamlit as st

from src.core.tag_manager import TagManager


def render_bulk_tag_actions(selected_documents: list[str]):
    """
    Renders the Bulk Tag Actions UI.

    Args:
        selected_documents (List[str]): List of document IDs (filenames) that are currently selected.
    """
    st.markdown("### Bulk Tag Actions")

    if not selected_documents:
        st.info("No documents selected.")
        return

    st.write(f"**Selected Documents:** {len(selected_documents)}")

    tag_input = st.text_input("Tag", placeholder="e.g. #graded_batch_1")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Apply Tag", use_container_width=True):
            if not tag_input.strip():
                st.warning("Please enter a tag.")
            else:
                TagManager.apply_tag(selected_documents, tag_input.strip())
                st.success(f"Tag applied to {len(selected_documents)} documents.")

    with col2:
        if st.button("Remove Tag", use_container_width=True):
            if not tag_input.strip():
                st.warning("Please enter a tag.")
            else:
                TagManager.remove_tag(selected_documents, tag_input.strip())
                st.success(f"Tag removed from {len(selected_documents)} documents.")
