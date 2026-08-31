import hashlib
import os

import streamlit as st
from streamlit.web.server.websocket_headers import _get_websocket_headers


def get_client_fingerprint() -> str:
    """
    Extracts browser context signatures securely from incoming WebSocket request headers.
    Falls back to environment attributes if headers are dropped by proxies.
    """
    headers = _get_websocket_headers() or {}

    # 1. Resolve Remote Client IP address (accounting for Reverse Proxy routing headers)
    ip_address = (
        headers.get("X-Forwarded-For", "").split(",")[0].strip()
        or headers.get("X-Real-Ip", "").strip()
        or "127.0.0.1"
    )

    # 2. Capture Browser User-Agent details
    user_agent = headers.get("User-Agent", "Unknown-Browser-Agent")

    return f"{ip_address}|{user_agent}"


def generate_secure_session_id(base_uuid: str) -> str:
    """
    Binds a standard UUID tracking token to a specific client network/browser fingerprint.
    Utilizes an environment pepper to prevent reverse-engineering of hashes.
    """
    fingerprint = get_client_fingerprint()
    secret_pepper = os.getenv("SESSION_SECRET_PEPPER", "fallback_secure_string_3442")

    # Construct a cryptographically bound tracking token
    bound_payload = f"{base_uuid}{fingerprint}{secret_pepper}"
    secure_hash = hashlib.sha256(bound_payload.encode("utf-8")).hexdigest()

    return secure_hash


def initialize_and_verify_session():
    """
    Validates the active session state structure against the incoming runtime signature.
    Resets tracking if fingerprint modifications are caught to prevent hijacking.
    """
    # 1. Fallback Generation if session tracking is empty
    if "raw_session_uuid" not in st.session_state:
        # Generate a secure random crypto token string for storage instead of a predictable UUID template
        st.session_state.raw_session_uuid = os.urandom(32).hex()

    # 2. Compute dynamic expected session validation key for current execution request context
    expected_secure_id = generate_secure_session_id(st.session_state.raw_session_uuid)

    # 3. Handle cross-session state injection validation
    if "active_session_token" in st.session_state:
        if st.session_state.active_session_token != expected_secure_id:
            # Drop context immediately to prevent hijacking attempts
            st.session_state.clear()
            st.error("🚨 Security Exception: Session verification signature mismatch.")
            st.stop()
    else:
        st.session_state.active_session_token = expected_secure_id

    # The verified key used as the storage bucket target inside your Redis Cache infrastructure
    return st.session_state.active_session_token
