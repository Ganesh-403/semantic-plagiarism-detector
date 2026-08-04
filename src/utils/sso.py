import os
import secrets

import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def get_google_auth_url() -> tuple[str, str]:
    """Return the Google OAuth authorization URL and state."""
    client_id = os.getenv("GOOGLE_CLIENT_ID")
    redirect_uri = os.getenv("APP_BASE_URL", "http://localhost:8501")
    state = secrets.token_urlsafe(16)

    url = (
        f"https://accounts.google.com/o/oauth2/v2/auth?"
        f"response_type=code&"
        f"client_id={client_id}&"
        f"redirect_uri={redirect_uri}&"
        f"scope=email%20profile&"
        f"state=google_{state}"
    )
    return url, f"google_{state}"


def exchange_google_code(code: str) -> dict | None:
    """Exchange code for access token and fetch user info."""
    client_id = os.getenv("GOOGLE_CLIENT_ID")
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
    redirect_uri = os.getenv("APP_BASE_URL", "http://localhost:8501")

    token_resp = requests.post(
        "https://oauth2.googleapis.com/token",
        data={
            "code": code,
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        },
    )
    if not token_resp.ok:
        return None

    access_token = token_resp.json().get("access_token")
    if not access_token:
        return None

    user_info_resp = requests.get(
        "https://www.googleapis.com/oauth2/v2/userinfo",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    if not user_info_resp.ok:
        return None

    return user_info_resp.json()


def get_github_auth_url() -> tuple[str, str]:
    """Return the GitHub OAuth authorization URL and state."""
    client_id = os.getenv("GITHUB_CLIENT_ID")
    redirect_uri = os.getenv("APP_BASE_URL", "http://localhost:8501")
    state = secrets.token_urlsafe(16)

    url = (
        f"https://github.com/login/oauth/authorize?"
        f"client_id={client_id}&"
        f"redirect_uri={redirect_uri}&"
        f"scope=user:email&"
        f"state=github_{state}"
    )
    return url, f"github_{state}"


def exchange_github_code(code: str) -> dict | None:
    """Exchange code for access token and fetch user info."""
    client_id = os.getenv("GITHUB_CLIENT_ID")
    client_secret = os.getenv("GITHUB_CLIENT_SECRET")
    redirect_uri = os.getenv("APP_BASE_URL", "http://localhost:8501")

    token_resp = requests.post(
        "https://github.com/login/oauth/access_token",
        data={
            "client_id": client_id,
            "client_secret": client_secret,
            "code": code,
            "redirect_uri": redirect_uri,
        },
        headers={"Accept": "application/json"},
    )
    if not token_resp.ok:
        return None

    access_token = token_resp.json().get("access_token")
    if not access_token:
        return None

    user_info_resp = requests.get(
        "https://api.github.com/user",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    if not user_info_resp.ok:
        return None

    user_data = user_info_resp.json()

    # GitHub might not return email in /user if it's private, fetch explicitly
    if not user_data.get("email"):
        emails_resp = requests.get(
            "https://api.github.com/user/emails",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        if emails_resp.ok:
            emails = emails_resp.json()
            primary_email = next((e["email"] for e in emails if e.get("primary")), None)
            if primary_email:
                user_data["email"] = primary_email

    return user_data
