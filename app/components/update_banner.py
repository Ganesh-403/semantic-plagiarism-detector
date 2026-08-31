"""Dismissible Streamlit banner for available app updates."""

import streamlit as st

_DISMISS_KEY = "update_banner_dismissed"


def render_update_banner(latest_version: str) -> None:
    """Render a dismissible banner announcing a newer available version."""
    if st.session_state.get(_DISMISS_KEY, False):
        return

    col1, col2 = st.columns([6, 1])
    with col1:
        st.info(
            f"🔔 A new version ({latest_version}) is available. "
            f"[View release notes](https://github.com/Ganesh-403/"
            f"semantic-plagiarism-detector/releases/tag/{latest_version})"
        )
    with col2:
        if st.button("Dismiss", key="dismiss_update_banner"):
            st.session_state[_DISMISS_KEY] = True
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()