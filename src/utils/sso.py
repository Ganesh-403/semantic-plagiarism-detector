import logging
import os
import secrets
import urllib.parse

import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def _load_env() -> None:
    """Lazily load environment variables from .env file."""
    load_dotenv()


def _get_redirect_uri() -> str:
    """Return the configured OAuth redirect URI."""
    return os.getenv("APP_BASE_URL", "http://localhost:8501")


def get_google_auth_url() -> tuple[str, str]:
    """Return the Google OAuth authorization URL and state."""
    _load_env()
    client_id = os.getenv("GOOGLE_CLIENT_ID")
    if not client_id:
        raise ValueError("GOOGLE_CLIENT_ID environment variable is not configured")
    redirect_uri = _get_redirect_uri()
    state = f"google_{secrets.token_urlsafe(16)}"

    query_params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": "email profile",
        "state": state,
    }

    encoded_args = urllib.parse.urlencode(query_params)
    url = f"https://accounts.google.com/o/oauth2/v2/auth?{encoded_args}"

    return url, state


def exchange_google_code(code: str) -> tuple[dict | None, str | None]:
    """Exchange code for access token and fetch user info."""
    _load_env()
    client_id = os.getenv("GOOGLE_CLIENT_ID")
    if not client_id:
        raise ValueError("GOOGLE_CLIENT_ID environment variable is not configured")
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
    if not client_secret:
        raise ValueError("GOOGLE_CLIENT_SECRET environment variable is not configured")
    redirect_uri = _get_redirect_uri()

    try:
        token_resp = requests.post(
            "https://oauth2.googleapis.com/token",
            data={
                "code": code,
                "client_id": client_id,
                "client_secret": client_secret,
                "redirect_uri": redirect_uri,
                "grant_type": "authorization_code",
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
    access_token = token_json.get("access_token")
    if not access_token:
        return None, "Invalid or expired SSO authorization code"

    try:
        user_info_resp = requests.get(
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

    return user_info_resp.json(), None


def get_github_auth_url() -> tuple[str, str]:
    """Return the GitHub OAuth authorization URL and state."""
    _load_env()
    client_id = os.getenv("GITHUB_CLIENT_ID")
    if not client_id:
        raise ValueError("GITHUB_CLIENT_ID environment variable is not configured")
    redirect_uri = _get_redirect_uri()
    state = f"github_{secrets.token_urlsafe(16)}"

    query_params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": "user:email",
        "state": state,
    }

    encoded_args = urllib.parse.urlencode(query_params)
    url = f"https://github.com/login/oauth/authorize?{encoded_args}"

    return url, state


def exchange_github_code(code: str) -> tuple[dict | None, str | None]:
    """Exchange code for access token and fetch user info."""
    _load_env()
    client_id = os.getenv("GITHUB_CLIENT_ID")
    if not client_id:
        raise ValueError("GITHUB_CLIENT_ID environment variable is not configured")
    client_secret = os.getenv("GITHUB_CLIENT_SECRET")
    if not client_secret:
        raise ValueError("GITHUB_CLIENT_SECRET environment variable is not configured")
    redirect_uri = _get_redirect_uri()

    try:
        token_resp = requests.post(
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
        logger.error(
            f"GitHub OAuth error response: {token_json.get('error_description') or token_json.get('error')}"
        )
        return None, "Invalid or expired SSO authorization code"

    access_token = token_json.get("access_token")
    if not access_token:
        return None, "Invalid or expired SSO authorization code"

    try:
        user_info_resp = requests.get(
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
    if user_data.get("email") and user_data["email"].endswith(
        "@users.noreply.github.com"
    ):
        user_data["email"] = None

    # GitHub might not return email in /user if it's private, fetch explicitly
    if not user_data.get("email"):
        try:
            emails_resp = requests.get(
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
                e
                for e in emails
                if e.get("verified")
                and not e["email"].endswith("@users.noreply.github.com")
            ]
            # Try primary first, then fallback to first available valid email
            primary_email = next(
                (e["email"] for e in valid_emails if e.get("primary")), None
            )
            if primary_email:
                user_data["email"] = primary_email
            elif valid_emails:
                user_data["email"] = valid_emails[0]["email"]
            else:
                user_data["email"] = None

    if not user_data.get("email"):
        # We raise a ValueError to reject login with message requesting a public email.
        raise ValueError(
            "GitHub login failed: A verified public email is required. Please update your GitHub settings."
        )

    return user_data, None
