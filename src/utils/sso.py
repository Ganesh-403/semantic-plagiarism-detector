# src/utils/sso.py

import base64
import hashlib
import logging
import os
import re
import secrets
import time
import urllib.parse
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# State expiration constant - 10 minutes
STATE_EXPIRATION_SECONDS = 600


def _get_oauth_session() -> requests.Session:
    """Create and return a requests.Session configured with retry logic for transient errors.

    Acceptance Criteria (Issue #3455):
    - Uses urllib3.util.Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
    """
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[500, 502, 503, 504],
        raise_on_status=False,  # Keep status code responses accessible so we can log/handle them.
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


@dataclass
class SSOUserProfile:
    email: str
    username: str
    name: str
    avatar: str


def _load_env() -> None:
    """Lazily load environment variables from .env file."""
    load_dotenv()


def _get_redirect_uri() -> str:
    """Return the configured OAuth redirect URI."""
    return os.getenv("APP_BASE_URL", "http://localhost:8501")


def verify_sso_state_payload(
    state: str, stored_state: Dict[str, Any]
) -> Tuple[bool, Optional[str]]:
    """
    Verify a state token against the ``state_data`` payload it was issued with.

    This is the stateless counterpart to :func:`verify_sso_state`. That one
    asks the auth database whether a state row is live and unused; this one
    checks a token against the dict returned as the third element of
    :func:`get_google_auth_url` / :func:`get_github_auth_url`, which is useful
    when the payload is held in the session rather than the database.

    Args:
        state: The state parameter received from the OAuth callback
        stored_state: The stored state data containing token and timestamp

    Returns:
        tuple[bool, Optional[str]]: (is_valid, error_message)
    """
    if not stored_state:
        return False, "Invalid state parameter"
    
    # Check if stored_state has the expected structure
    if not isinstance(stored_state, dict):
        return False, "Invalid state data format"
    
    # Get the state token and timestamp
    stored_token = stored_state.get("token")
    if not stored_token:
        return False, "Invalid state data: missing token"
    
    # Verify the state token matches
    if state != stored_token:
        return False, "Invalid state token"
    
    # Check expiration
    created_at = stored_state.get("created_at")
    if not created_at:
        # If no timestamp, treat as invalid for security
        return False, "Invalid state data: missing timestamp"
    
    # Handle both string and integer timestamps
    if isinstance(created_at, str):
        try:
            created_at = float(created_at)
        except ValueError:
            return False, "Invalid state timestamp format"
    elif not isinstance(created_at, (int, float)):
        return False, "Invalid state timestamp type"
    
    # Check if state has expired
    current_time = time.time()
    elapsed_seconds = current_time - created_at
    
    if elapsed_seconds > STATE_EXPIRATION_SECONDS:
        return False, f"State token expired (elapsed: {elapsed_seconds:.0f}s, max: {STATE_EXPIRATION_SECONDS}s)"
    
    return True, None


def generate_pkce_pair() -> Tuple[str, str]:
    """Generate a PKCE code_verifier and code_challenge pair.

    Uses ``secrets.token_urlsafe(32)`` for the verifier and SHA-256 with
    base64url encoding (no padding) for the challenge, as specified by
    RFC 7636.

    Returns:
        tuple[str, str]: (code_verifier, code_challenge)
    """
    code_verifier = secrets.token_urlsafe(32)
    digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
    code_challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return code_verifier, code_challenge


def get_google_auth_url() -> Tuple[str, str, Dict[str, Any]]:
    """
    Return the Google OAuth authorization URL, state, and state data.

    PKCE (Issue #3453): The returned ``state_data`` dict includes a
    ``code_verifier`` key that **must** be passed to
    :func:`exchange_google_code` when exchanging the authorization code.

    Returns:
        tuple[str, str, dict]: (authorization_url, state_token, state_data)
    """
    _load_env()
    client_id = os.getenv("GOOGLE_CLIENT_ID")
    if not client_id:
        raise ValueError("GOOGLE_CLIENT_ID environment variable is not configured")
    
    redirect_uri = _get_redirect_uri()
    state = f"google_{secrets.token_urlsafe(16)}"

    # PKCE: generate code_verifier / code_challenge pair (Issue #3453)
    code_verifier, code_challenge = generate_pkce_pair()
    
    # Create state data with timestamp for expiration checking
    state_data = {
        "token": state,
        "created_at": time.time(),
        "provider": "google",
        "code_verifier": code_verifier,
    }

    try:
        from src.db.auth import store_sso_state
        store_sso_state(state)
    except Exception as e:
        logger.warning(f"Failed to store Google SSO state parameter: {e}")

    google_scopes = os.getenv("GOOGLE_OAUTH_SCOPES", "email profile")

    query_params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": google_scopes,
        "state": state,
        "prompt": "select_account",
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    }

    encoded_args = urllib.parse.urlencode(query_params)
    url = f"https://accounts.google.com/o/oauth2/v2/auth?{encoded_args}"

    return url, state, state_data


def exchange_google_code(code: str, state: str | None = None, code_verifier: str | None = None) -> tuple[SSOUserProfile | None, str | None]:
    """Exchange code for access token and fetch user info.

    Args:
        code: The authorization code from the OAuth callback.
        state: Optional CSRF state token for validation.
        code_verifier: Optional PKCE code_verifier (Issue #3453).  When
            supplied, it is included in the token exchange request so that
            Google can verify the proof key.
    """
    if state is not None:
        if not verify_sso_state(state):
            return None, "Invalid or expired SSO state parameter (CSRF protection failed)."

    _load_env()
    client_id = os.getenv("GOOGLE_CLIENT_ID")
    if not client_id:
        raise ValueError("GOOGLE_CLIENT_ID environment variable is not configured")
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
    if not client_secret:
        raise ValueError("GOOGLE_CLIENT_SECRET environment variable is not configured")
    redirect_uri = _get_redirect_uri()

    token_data = {
        "code": code,
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": redirect_uri,
        "grant_type": "authorization_code",
    }
    # PKCE: include code_verifier in token exchange (Issue #3453)
    if code_verifier is not None:
        token_data["code_verifier"] = code_verifier

    # Setup retrying OAuth session
    session = _get_oauth_session()

    try:
        token_resp = session.post(
            "https://oauth2.googleapis.com/token",
            data=token_data,
            timeout=10,
        )
    except requests.Timeout:
        logger.error("OAuth token exchange timed out")
        return None, "SSO provider timed out. Please try again."
    except Exception as e:
        logger.error(f"OAuth token exchange unexpected error: {e}")
        return None, "SSO authentication failed"

    if 400 <= token_resp.status_code < 500:
        return None, "Invalid or expired SSO authorization code"
    if not token_resp.ok:
        return None, "SSO authentication failed"

    token_json = token_resp.json()
    access_token = token_json.get("access_token")
    if not access_token:
        return None, "Invalid or expired SSO authorization code"

    try:
        user_info_resp = session.get(
            "https://www.googleapis.com/oauth2/v2/userinfo",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=10,
        )
    except requests.Timeout:
        logger.error("OAuth user information request timed out")
        return None, "SSO provider timed out. Please try again."
    except Exception as e:
        logger.error(f"OAuth user information request unexpected error: {e}")
        return None, "SSO authentication failed"

    if 400 <= user_info_resp.status_code < 500:
        return None, "Invalid or expired SSO authorization code"
    if not user_info_resp.ok:
        return None, "SSO authentication failed"

    user_data = user_info_resp.json()
    email = user_data.get("email", "")
    raw_username = email.split("@")[0] if email else ""
    username = re.sub(r"[^a-zA-Z0-9_-]", "_", raw_username)

    # Fallback avatar generation (Issue #3459)
    avatar_url = user_data.get("picture")
    if not avatar_url:
        name_param = urllib.parse.quote(user_data.get("name") or raw_username or email)
        avatar_url = f"https://ui-avatars.com/api/?name={name_param}"

    profile = SSOUserProfile(
        email=email,
        username=username,
        name=user_data.get("name", ""),
        avatar=avatar_url
    )
    return profile, None


def verify_sso_state(state: str) -> bool:
    """Verify that the state parameter returned in the OAuth callback matches the stored value.

    Checks the token against the ``sso_states`` table via
    :func:`src.db.auth.validate_sso_state`, which also consumes the row so a
    replayed callback is rejected. See :func:`verify_sso_state_payload` for the
    stateless equivalent that validates a session-held ``state_data`` dict.

    Args:
        state: State token string returned from OAuth provider callback.

    Returns:
        bool: True if state is valid, unexpired, and not previously used; False otherwise.
    """
    if not state:
        logger.warning("SSO state verification failed: Empty state parameter.")
        try:
            from src.db.auth import log_security_event
            log_security_event("SSO_CSRF_REJECTED", username="anonymous", details=f"Invalid state: {state}")
        except Exception as audit_err:
            logger.warning(f"Failed to log security event for SSO CSRF rejection: {audit_err}")
        return False

    try:
        from src.db.auth import log_security_event, validate_sso_state
        is_valid = validate_sso_state(state)
        if not is_valid:
            logger.warning(f"CSRF protection: Invalid or expired SSO state parameter '{state}'")
            try:
                log_security_event("SSO_CSRF_REJECTED", username="anonymous", details=f"Invalid state: {state}")
            except Exception as audit_err:
                logger.warning(f"Failed to log security event for SSO CSRF rejection: {audit_err}")
        return is_valid
    except Exception as e:
        logger.error(f"Error during SSO state verification: {e}")
        try:
            from src.db.auth import log_security_event
            log_security_event("SSO_CSRF_REJECTED", username="anonymous", details=f"Invalid state: {state}")
        except Exception as audit_err:
            logger.warning(f"Failed to log security event for SSO CSRF rejection: {audit_err}")
        return False


def get_github_auth_url() -> Tuple[str, str, Dict[str, Any]]:
    """
    Return the GitHub OAuth authorization URL, state, and state data.
    
    Returns:
        tuple[str, str, dict]: (authorization_url, state_token, state_data)
    """
    _load_env()
    client_id = os.getenv("GITHUB_CLIENT_ID")
    if not client_id:
        raise ValueError("GITHUB_CLIENT_ID environment variable is not configured")
    
    redirect_uri = _get_redirect_uri()
    state = f"github_{secrets.token_urlsafe(16)}"
    
    # Create state data with timestamp for expiration checking
    state_data = {
        "token": state,
        "created_at": time.time(),
        "provider": "github"
    }

    try:
        from src.db.auth import store_sso_state
        store_sso_state(state)
    except Exception as e:
        logger.warning(f"Failed to store GitHub SSO state parameter: {e}")

    github_scopes = os.getenv("GITHUB_OAUTH_SCOPES", "user:email")

    query_params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": github_scopes,
        "state": state,
    }

    encoded_args = urllib.parse.urlencode(query_params)
    url = f"https://github.com/login/oauth/authorize?{encoded_args}"

    return url, state, state_data


def exchange_github_code(code: str, state: str | None = None) -> tuple[SSOUserProfile | None, str | None]:
    """Exchange code for access token and fetch user info."""
    if state is not None:
        if not verify_sso_state(state):
            return None, "Invalid or expired SSO state parameter (CSRF protection failed)."

    _load_env()
    client_id = os.getenv("GITHUB_CLIENT_ID")
    if not client_id:
        raise ValueError("GITHUB_CLIENT_ID environment variable is not configured")
    client_secret = os.getenv("GITHUB_CLIENT_SECRET")
    if not client_secret:
        raise ValueError("GOOGLE_CLIENT_SECRET environment variable is not configured")
    redirect_uri = _get_redirect_uri()

    # Setup retrying OAuth session
    session = _get_oauth_session()

    try:
        token_resp = session.post(
            "https://github.com/login/oauth/access_token",
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "code": code,
                "redirect_uri": redirect_uri,
            },
            headers={"Accept": "application/json"},
            timeout=10,
        )
    except requests.Timeout:
        logger.error("OAuth token exchange timed out")
        return None, "SSO provider timed out. Please try again."
    except Exception as e:
        logger.error(f"OAuth token exchange unexpected error: {e}")
        return None, "SSO authentication failed"

    if 400 <= token_resp.status_code < 500:
        return None, "Invalid or expired SSO authorization code"
    if not token_resp.ok:
        return None, "SSO authentication failed"

    token_json = token_resp.json()
    if token_json.get("error"):
        logger.error(f"GitHub OAuth error response: {token_json.get('error_description') or token_json.get('error')}")
        return None, "Invalid or expired SSO authorization code"

    access_token = token_json.get("access_token")
    if not access_token:
        return None, "Invalid or expired SSO authorization code"

    try:
        user_info_resp = session.get(
            "https://api.github.com/user",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=10,
        )
    except requests.Timeout:
        logger.error("OAuth user information request timed out")
        return None, "SSO provider timed out. Please try again."
    except Exception as e:
        logger.error(f"OAuth user information request unexpected error: {e}")
        return None, "SSO authentication failed"

    if 400 <= user_info_resp.status_code < 500:
        return None, "Invalid or expired SSO authorization code"
    if not user_info_resp.ok:
        return None, "SSO authentication failed"

    user_data = user_info_resp.json()

    # Filter out users.noreply.github.com email in user profile info
    if user_data.get("email") and user_data["email"].endswith("@users.noreply.github.com"):
        user_data["email"] = None

    # GitHub might not return email in /user if it's private, fetch explicitly
    if not user_data.get("email"):
        try:
            emails_resp = session.get(
                "https://api.github.com/user/emails",
                headers={"Authorization": f"Bearer {access_token}"},
                timeout=10,
            )
        except requests.Timeout:
            logger.error("OAuth user emails request timed out")
            return None, "SSO provider timed out. Please try again."
        except Exception as e:
            logger.error(f"OAuth user emails request unexpected error: {e}")
            emails_resp = None

        if emails_resp and 400 <= emails_resp.status_code < 500:
            return None, "Invalid or expired SSO authorization code"

        if emails_resp and emails_resp.ok:
            emails = emails_resp.json()
            # Filter emails: must be verified and not a noreply address
            valid_emails = [
                e for e in emails 
                if e.get("verified") and not e["email"].endswith("@users.noreply.github.com")
            ]
            # Try primary first, then fallback to first available valid email
            primary_email = next((e["email"] for e in valid_emails if e.get("primary")), None)
            if primary_email:
                user_data["email"] = primary_email
            elif valid_emails:
                user_data["email"] = valid_emails[0]["email"]
            else:
                user_data["email"] = None

    if not user_data.get("email"):
        # We raise a ValueError to reject login with message requesting a public email.
        raise ValueError("GitHub login failed: A verified public email is required. Please update your GitHub settings.")

    # Fallback avatar generation (Issue #3459)
    avatar_url = user_data.get("avatar_url")
    if not avatar_url:
        name_param = urllib.parse.quote(user_data.get("name") or user_data.get("login") or user_data["email"])
        avatar_url = f"https://ui-avatars.com/api/?name={name_param}"

    profile = SSOUserProfile(
        email=user_data["email"],
        username=user_data.get("login", ""),
        name=user_data.get("name", ""),
        avatar=avatar_url
    )
    return profile, None


def create_state_token(provider: str) -> Tuple[str, Dict[str, Any]]:
    """
    Create a new state token with timestamp.
    
    Args:
        provider: The OAuth provider ("google" or "github")
    
    Returns:
        tuple[str, dict]: (state_token, state_data)
    """
    token = f"{provider}_{secrets.token_urlsafe(16)}"
    state_data = {
        "token": token,
        "created_at": time.time(),
        "provider": provider
    }
    return token, state_data


def cleanup_expired_states(states: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Remove expired state tokens from memory dictionary.
    
    Args:
        states: Dictionary mapping state tokens to state metadata
        
    Returns:
        dict: Cleaned states dictionary
    """
    current_time = time.time()
    expiration_threshold = current_time - STATE_EXPIRATION_SECONDS
    
    # Filter out expired states
    valid_states = {
        token: data for token, data in states.items()
        if data.get("created_at", 0) > expiration_threshold
    }
    
    expired_count = len(states) - len(valid_states)
    if expired_count > 0:
        logger.info(f"Cleaned up {expired_count} expired OAuth state tokens")
    
    return valid_states


def get_azure_auth_url() -> tuple[str, str]:
    """Return the Microsoft / Azure AD OAuth authorization URL and state."""
    _load_env()
    client_id = os.getenv("AZURE_CLIENT_ID")
    if not client_id:
        raise ValueError("AZURE_CLIENT_ID environment variable is not configured")
    tenant_id = os.getenv("AZURE_TENANT_ID", "common")
    redirect_uri = _get_redirect_uri()
    state = f"azure_{secrets.token_urlsafe(16)}"

    try:
        from src.db.auth import store_sso_state
        store_sso_state(state)
    except Exception as e:
        logger.warning(f"Failed to store Azure SSO state parameter: {e}")

    query_params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": "openid profile email User.Read",
        "state": state,
        "prompt": "select_account",
    }

    encoded_args = urllib.parse.urlencode(query_params)
    url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/authorize?{encoded_args}"

    return url, state


def exchange_azure_code(code: str, state: str | None = None) -> tuple[SSOUserProfile | None, str | None]:
    """Exchange Azure AD authorization code for access token and fetch user info."""
    if state is not None:
        if not verify_sso_state(state):
            return None, "Invalid or expired SSO state parameter (CSRF protection failed)."

    _load_env()
    client_id = os.getenv("AZURE_CLIENT_ID")
    if not client_id:
        raise ValueError("AZURE_CLIENT_ID environment variable is not configured")
    client_secret = os.getenv("AZURE_CLIENT_SECRET")
    if not client_secret:
        raise ValueError("AZURE_CLIENT_SECRET environment variable is not configured")
    tenant_id = os.getenv("AZURE_TENANT_ID", "common")
    redirect_uri = _get_redirect_uri()

    try:
        token_resp = requests.post(
            f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token",
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "code": code,
                "redirect_uri": redirect_uri,
                "grant_type": "authorization_code",
                "scope": "openid profile email User.Read",
            },
            timeout=10,
        )
    except requests.Timeout:
        logger.error("OAuth token exchange timed out")
        return None, "SSO provider timed out. Please try again."
    except Exception as e:
        logger.error(f"OAuth token exchange unexpected error: {e}")
        return None, "SSO authentication failed"

    if 400 <= token_resp.status_code < 500:
        return None, "Invalid or expired SSO authorization code"
    if not token_resp.ok:
        return None, "SSO authentication failed"

    token_json = token_resp.json()
    if token_json.get("error"):
        logger.error(f"Azure OAuth error response: {token_json.get('error_description') or token_json.get('error')}")
        return None, "Invalid or expired SSO authorization code"

    access_token = token_json.get("access_token")
    if not access_token:
        return None, "Invalid or expired SSO authorization code"

    try:
        user_info_resp = requests.get(
            "https://graph.microsoft.com/v1.0/me",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=10,
        )
    except requests.Timeout:
        logger.error("OAuth user information request timed out")
        return None, "SSO provider timed out. Please try again."
    except Exception as e:
        logger.error(f"OAuth user information request unexpected error: {e}")
        return None, "SSO authentication failed"

    if 400 <= user_info_resp.status_code < 500:
        return None, "Invalid or expired SSO authorization code"
    if not user_info_resp.ok:
        return None, "SSO authentication failed"

    user_data = user_info_resp.json()
    email = user_data.get("mail") or user_data.get("userPrincipalName", "")
    raw_username = email.split("@")[0] if email else ""
    username = re.sub(r"[^a-zA-Z0-9_-]", "_", raw_username)

    profile = SSOUserProfile(
        email=email,
        username=username,
        name=user_data.get("displayName", ""),
        avatar="",
    )
    return profile, None
