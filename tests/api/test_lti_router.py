"""LTI launch signatures, browser binding, and registration validation."""

import time
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient
from src.api.app import app
from src.api.routers import lti_router as lti

PREFIX = "https://purl.imsglobal.org/spec/lti/claim/"


@pytest.fixture
def platform(monkeypatch, tmp_path):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    monkeypatch.setenv("LTI_PRIVATE_KEY_PATH", str(tmp_path / "tool.pem"))
    monkeypatch.setattr(lti, "_oidc_states", {})
    monkeypatch.setattr(lti, "LMS_ISSUER", "https://lms.example")
    monkeypatch.setattr(lti, "LMS_CLIENT_ID", "tool-client")
    monkeypatch.setattr(lti, "LTI_DEPLOYMENT_ID", "deployment")
    monkeypatch.setattr(lti, "_platform_keys", lambda url: SimpleNamespace(
        get_signing_key_from_jwt=lambda token: SimpleNamespace(key=key.public_key())
    ))
    with TestClient(app) as client:
        yield client, key


def begin_launch(client):
    response = client.get("/api/v1/lti/login", params={"iss": lti.LMS_ISSUER, "login_hint": "user"}, follow_redirects=False)
    assert response.status_code == 303
    params = parse_qs(urlsplit(response.headers["location"]).query)
    now = int(time.time())
    claims = {
        "iss": lti.LMS_ISSUER, "aud": lti.LMS_CLIENT_ID, "sub": "student",
        "iat": now, "exp": now + 300, "nonce": params["nonce"][0],
        PREFIX + "deployment_id": lti.LTI_DEPLOYMENT_ID,
        PREFIX + "version": "1.3.0", PREFIX + "message_type": "LtiResourceLinkRequest",
    }
    return params["state"][0], claims


def launch(client, state, claims, key):
    token = jwt.encode(claims, key, algorithm="RS256", headers={"kid": "platform-key"})
    return client.post("/api/v1/lti/launch", data={"state": state, "id_token": token}, follow_redirects=False)


def test_jwks_endpoint(platform):
    client, _ = platform
    response = client.get("/api/v1/lti/jwks")
    assert response.status_code == 200
    key = response.json()["keys"][0]
    assert key["kty"] == "RSA" and key["alg"] == "RS256"
    assert "d" not in key  # Public material only.


def test_login_init_endpoint(platform):
    client, _ = platform
    state, claims = begin_launch(client)
    assert state != claims["nonce"]
    assert client.cookies.get("lti_state") == state


def test_login_accepts_form_post(platform):
    client, _ = platform
    response = client.post("/api/v1/lti/login", data={"iss": lti.LMS_ISSUER, "login_hint": "user"}, follow_redirects=False)
    assert response.status_code == 303


def test_login_rejects_unknown_issuer(platform):
    client, _ = platform
    response = client.get("/api/v1/lti/login?iss=unknown&login_hint=user")
    assert response.status_code == 400
    assert not lti._oidc_states


def test_launch_invalid_state(platform):
    client, _ = platform
    response = client.post("/api/v1/lti/launch", data={"state": "invalid", "id_token": "fake"})
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid state"


def test_signed_launch_and_replay_rejection(platform):
    client, key = platform
    state, claims = begin_launch(client)
    assert launch(client, state, claims, key).status_code == 307
    assert launch(client, state, claims, key).status_code == 400


@pytest.mark.parametrize(("claim", "value"), [
    ("iss", "https://attacker.example"), ("aud", "another-tool"),
    ("exp", 1), ("nonce", "wrong"), ("azp", "wrong"),
    (PREFIX + "deployment_id", "wrong"), (PREFIX + "version", "1.1"),
    (PREFIX + "message_type", "unknown"),
])
def test_launch_rejects_invalid_claims(platform, claim, value):
    client, key = platform
    state, claims = begin_launch(client)
    claims[claim] = value
    assert launch(client, state, claims, key).status_code == 400


def test_launch_rejects_forged_signature(platform):
    client, _ = platform
    state, claims = begin_launch(client)
    attacker_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    assert launch(client, state, claims, attacker_key).status_code == 400


def test_launch_requires_browser_cookie(platform):
    client, key = platform
    state, claims = begin_launch(client)
    client.cookies.clear()
    assert launch(client, state, claims, key).status_code == 400


def test_launch_rejects_expired_state(platform):
    client, key = platform
    state, claims = begin_launch(client)
    lti._oidc_states[state]["expires_at"] = 0
    assert launch(client, state, claims, key).status_code == 400


def test_score_endpoint_remains_private(platform):
    client, _ = platform
    response = client.post("/api/v1/lti/scores?lms_lineitem_url=https://lms.example/line", json={"user_id": "student", "score": 1})
    assert response.status_code == 401


def test_signed_deep_link_response_escapes_return_url(platform):
    client, key = platform
    state, claims = begin_launch(client)
    claims[PREFIX + "message_type"] = "LtiDeepLinkingRequest"
    claims["https://purl.imsglobal.org/spec/lti-dl/claim/deep_linking_settings"] = {
        "deep_link_return_url": "https://lms.example/return?x=1&y=2"
    }
    response = launch(client, state, claims, key)
    assert response.status_code == 200
    assert 'action="https://lms.example/return?x=1&amp;y=2"' in response.text
    assert 'name="JWT"' in response.text


def test_signed_deep_link_rejects_unregistered_return_origin(platform):
    client, key = platform
    state, claims = begin_launch(client)
    claims[PREFIX + "message_type"] = "LtiDeepLinkingRequest"
    claims["https://purl.imsglobal.org/spec/lti-dl/claim/deep_linking_settings"] = {
        "deep_link_return_url": "https://attacker.example/return"
    }
    assert launch(client, state, claims, key).status_code == 400
