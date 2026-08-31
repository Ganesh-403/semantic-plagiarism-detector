"""
Authentication & Login View Component.

Renders OAuth SSO callback handling, 2FA verification forms, and username/password login.
"""

import time

import pyotp
import streamlit as st

from app.session_keys import SessionKeys
from app.theme import set_theme
from src.core.config import DEFAULT_THRESHOLDS
from src.db.auth import (
    authenticate_user,
    get_2fa_status,
    get_or_create_sso_user,
    get_user_preferences,
    get_user_role,
    is_user_active,
)
from src.utils.redis_cache import cache_session_state
from src.utils.sso import (
    SSOConfigurationError,
    exchange_github_code,
    exchange_google_code,
)


def handle_oauth_callbacks(session_id: str):
    """Handle incoming OAuth query parameter callbacks from Google / GitHub SSO."""
    if not st.session_state.get(SessionKeys.AUTHENTICATED, False):
        if "code" in st.query_params and "state" in st.query_params:
            _code = st.query_params["code"]
            _state = st.query_params["state"]

            _user_info, _error_msg = None, None
            try:
                if _state.startswith("google_"):
                    _user_info, _error_msg = exchange_google_code(_code)
                elif _state.startswith("github_"):
                    _user_info, _error_msg = exchange_github_code(_code)
            except (SSOConfigurationError, ValueError) as _exc:
                _user_info, _error_msg = None, f"Configuration Error: {_exc}"

            if _user_info and _user_info.get("email"):
                _email = _user_info["email"]
                if not is_user_active(_email):
                    st.error("🚨 Account suspended. Please contact your administrator.")
                    st.query_params.clear()
                else:
                    _role = get_or_create_sso_user(_email)
                    st.session_state[SessionKeys.AUTHENTICATED] = True
                    st.session_state[SessionKeys.USERNAME] = _email
                    st.session_state[SessionKeys.ROLE] = _role
                    st.session_state[SessionKeys.LAST_INTERACTION] = time.time()
                    cache_session_state(session_id, SessionKeys.AUTHENTICATED, True)
                    cache_session_state(session_id, SessionKeys.USERNAME, _email)
                    cache_session_state(session_id, SessionKeys.ROLE, _role)
                    cache_session_state(
                        session_id, SessionKeys.LAST_INTERACTION, time.time()
                    )
                    st.query_params.clear()
                    st.rerun()
            else:
                _err = _error_msg or "Could not retrieve your email."
                st.error(f"🚨 SSO authentication failed: {_err}")
                st.query_params.clear()


def render_login_view(session_id: str):
    """Render login form and 2FA verification prompt."""
    if st.session_state.get(SessionKeys.PENDING_2FA, False):
        with st.form("otp_form"):
            st.subheader("🔒 Two-Factor Authentication")
            st.info("Enter the 6-digit verification token from your authenticator app.")
            otp_code = st.text_input(
                "Verification Code", max_chars=6, key="login_otp_code"
            )
            col1, col2 = st.columns(2)
            with col1:
                verify_submitted = st.form_submit_button(
                    "Verify", use_container_width=True
                )
            with col2:
                cancel_submitted = st.form_submit_button(
                    "Cancel", use_container_width=True
                )

            if verify_submitted:
                username = st.session_state.get(SessionKeys.PENDING_USERNAME)
                enabled, otp_secret = get_2fa_status(username)
                if enabled and otp_secret:
                    totp = pyotp.TOTP(otp_secret)
                    if totp.verify(otp_code.strip()):
                        role = st.session_state.get(SessionKeys.PENDING_ROLE)
                        st.session_state[SessionKeys.AUTHENTICATED] = True
                        st.session_state[SessionKeys.USERNAME] = username
                        st.session_state[SessionKeys.ROLE] = role
                        st.session_state[SessionKeys.LAST_INTERACTION] = time.time()

                        cache_session_state(session_id, SessionKeys.AUTHENTICATED, True)
                        cache_session_state(session_id, SessionKeys.USERNAME, username)
                        cache_session_state(session_id, SessionKeys.ROLE, role)
                        cache_session_state(
                            session_id, SessionKeys.LAST_INTERACTION, time.time()
                        )
                        prefs = get_user_preferences(username)
                        st.session_state.threshold = prefs.get(
                            "threshold", DEFAULT_THRESHOLDS.plagiarism
                        )
                        st.session_state.theme = prefs.get("theme", "Light")
                        set_theme(st.session_state.theme)

                        del st.session_state[SessionKeys.PENDING_2FA]
                        del st.session_state[SessionKeys.PENDING_USERNAME]
                        del st.session_state[SessionKeys.PENDING_ROLE]

                        st.success(f"✅ Welcome back, {username}!")
                        st.rerun()
                    else:
                        st.error("🚨 Invalid verification code. Please try again.")
                else:
                    st.error("🚨 2FA configuration error. Please contact admin.")

            if cancel_submitted:
                del st.session_state[SessionKeys.PENDING_2FA]
                del st.session_state[SessionKeys.PENDING_USERNAME]
                del st.session_state[SessionKeys.PENDING_ROLE]
                st.rerun()
            st.stop()

    st.header("🔑 Login")
    username_input = st.text_input("Username")
    password_input = st.text_input("Password", type="password")

    if st.button("Login"):
        if authenticate_user(username_input, password_input):
            role = get_user_role(username_input)
            enabled, _ = get_2fa_status(username_input)
            if enabled:
                st.session_state[SessionKeys.PENDING_2FA] = True
                st.session_state[SessionKeys.PENDING_USERNAME] = username_input
                st.session_state[SessionKeys.PENDING_ROLE] = role
                st.rerun()
            else:
                st.session_state[SessionKeys.AUTHENTICATED] = True
                st.session_state[SessionKeys.USERNAME] = username_input
                st.session_state[SessionKeys.ROLE] = role
                st.session_state[SessionKeys.LAST_INTERACTION] = time.time()
                cache_session_state(session_id, SessionKeys.AUTHENTICATED, True)
                cache_session_state(session_id, SessionKeys.USERNAME, username_input)
                cache_session_state(session_id, SessionKeys.ROLE, role)
                cache_session_state(
                    session_id, SessionKeys.LAST_INTERACTION, time.time()
                )
                st.rerun()
        else:
            st.error("Invalid username or password.")
    st.stop()
