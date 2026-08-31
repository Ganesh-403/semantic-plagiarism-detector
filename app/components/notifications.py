"""Simple Streamlit toast helper with a safe fallback."""

import streamlit as st

_ICONS = {
    "success": "✅",
    "warning": "⚠️",
    "error": "❌",
    "info": "ℹ️",
}


def show_notification(message: str, type: str = "info") -> None:
    """Show a toast notification, falling back to st.info if toast fails."""
    icon = _ICONS.get(type, _ICONS["info"])
    try:
        st.toast(message, icon=icon)
    except Exception:
        st.info(message)
