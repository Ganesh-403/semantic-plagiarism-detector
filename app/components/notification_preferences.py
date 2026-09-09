"""Persist the signed-in user's notification choices."""

import streamlit as st

from src.db.auth import get_notification_preferences, update_notification_preferences


def render_notification_preferences():
    username = st.session_state.get("username")
    if not st.session_state.get("authenticated") or not username:
        return
    st.markdown("### 🔔 Notification Preferences")
    preferences = get_notification_preferences(username)
    email = st.toggle(
        "📧 Email notifications",
        value=preferences["email_notifications"],
        key=f"email_notifications_{username}",
    )
    webhook = st.toggle(
        "🔗 Webhook notifications",
        value=preferences["webhook_notifications"],
        key=f"webhook_notifications_{username}",
    )
    if st.button("Save notification preferences", key="save_notification_preferences"):
        update_notification_preferences(username, email, webhook)
        st.success("Notification preferences saved.")
