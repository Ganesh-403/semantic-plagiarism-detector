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
