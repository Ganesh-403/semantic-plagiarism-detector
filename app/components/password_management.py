"""Password recovery and authenticated password rotation forms."""
import time
import streamlit as st
from src.db import auth
from src.security import password_recovery
from src.utils.redis_cache import clear_session


def _sign_out_after_change():
    clear_session(st.session_state.get("session_id", ""))
    st.session_state.clear()
    st.session_state["password_changed_notice"] = True
    st.rerun()


def render_password_change(required=False):
    username = st.session_state.get("username")
    if not st.session_state.get("authenticated") or not username:
        return
    if required:
        st.warning("Change your password before continuing.")
    with st.form("change_own_password", clear_on_submit=True):
        st.subheader("Change password")
        current = st.text_input("Current password", type="password")
        new = st.text_input("New password", type="password")
        confirm = st.text_input("Confirm new password", type="password")
        submitted = st.form_submit_button("Change password")
    if submitted:
        if new != confirm:
            st.error("New passwords do not match.")
        else:
            try:
                auth.update_password(username, new, current_user=username, old_password=current)
            except (ValueError, PermissionError) as exc:
                st.error(str(exc))
            else:
                _sign_out_after_change()
    if required:
        st.stop()


def enforce_password_rotation():
    username = st.session_state.get("username")
    if not username or not st.session_state.get("authenticated"):
        return
    with auth._connect() as connection:
        row = connection.execute("SELECT must_change_password, password_changed_at, status, role FROM users WHERE username = ?", (username,)).fetchone()
    if not row or row[2] != "active":
        clear_session(st.session_state.get("session_id", ""))
        st.session_state.clear()
        st.rerun()
    st.session_state["role"] = row[3]
    if row[1] and st.session_state.get("authenticated_at"):
        from datetime import datetime, timezone
        changed = datetime.fromisoformat(row[1])
        if changed.tzinfo is None:
            changed = changed.replace(tzinfo=timezone.utc)
        if changed.timestamp() > st.session_state["authenticated_at"]:
            _sign_out_after_change()
    if row and (row[0] or auth.is_password_expired(username)):
        render_password_change(required=True)


def render_password_recovery():
    with st.expander("Password recovery"):
        st.caption("Email recovery applies to accounts whose username is their email address. For other accounts, contact your administrator.")
        if password_recovery.mail_is_configured():
            with st.form("request_password_reset", clear_on_submit=True):
                email = st.text_input("Account email")
                request_reset = st.form_submit_button("Email reset instructions")
            if request_reset:
                from pydantic import EmailStr, TypeAdapter, ValidationError
                try:
                    email = str(TypeAdapter(EmailStr).validate_python(email))
                except ValidationError:
                    st.error("Enter a valid email address.")
                else:
                    if time.time() - st.session_state.get("last_recovery_request", 0) >= 60:
                        st.session_state["last_recovery_request"] = time.time()
                        import threading
                        threading.Thread(target=password_recovery.request_password_reset, args=(email,), daemon=True).start()
                    st.info(password_recovery.RESET_MESSAGE)
        else:
            st.info("Email recovery is not configured. Contact your administrator.")
        with st.form("complete_password_reset", clear_on_submit=True):
            token = st.text_input("Reset token", type="password")
            new = st.text_input("Recovery new password", type="password")
            confirm = st.text_input("Confirm recovery password", type="password")
            complete = st.form_submit_button("Reset password")
        if complete:
            if new != confirm:
                st.error("New passwords do not match.")
            else:
                try:
                    password_recovery.reset_password(token, new)
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    st.success("Password updated. Sign in with your new password.")
