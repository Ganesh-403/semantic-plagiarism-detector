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
document_filter.py
------------------
Document search filter component with Clear Search action button.
"""

from typing import Any

import pandas as pd
import streamlit as st


def render_document_filter(
    df_or_docs: Any,
    search_key: str = "document_search_query",
    placeholder: str = "Search documents by filename...",
) -> Any:
    """Render document search filter text input with an adjacent Clear Search button.

    Args:
        df_or_docs: pandas DataFrame or list of document records/dicts.
        search_key: Streamlit session state key for text input.
        placeholder: Text input placeholder.

    Returns:
        Filtered DataFrame or list of documents (or full view if search query is empty).
    """
    col1, col2 = st.columns([4, 1])

    with col1:
        st.text_input(
            "Search Documents",
            value=st.session_state.get(search_key, ""),
            placeholder=placeholder,
            key=search_key,
        )

    with col2:
        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
        if st.button(
            "Clear Search", key=f"clear_{search_key}_btn", use_container_width=True
        ):
            st.session_state[search_key] = ""
            st.rerun()

    query_str = (st.session_state.get(search_key, "") or "").strip().lower()
    if not query_str:
        return df_or_docs

    if isinstance(df_or_docs, pd.DataFrame):
        if df_or_docs.empty:
            return df_or_docs
        mask = df_or_docs.astype(str).apply(
            lambda row: row.str.lower().str.contains(query_str, regex=False).any(),
            axis=1,
        )
        return df_or_docs[mask]

    if isinstance(df_or_docs, list):
        filtered = []
        for doc in df_or_docs:
            fn = (
                doc.filename
                if hasattr(doc, "filename")
                else (doc.get("filename") if isinstance(doc, dict) else str(doc))
            )
            if query_str in str(fn).lower():
                filtered.append(doc)
        return filtered

    return df_or_docs
