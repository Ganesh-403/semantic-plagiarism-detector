import pytest
from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)

def test_jwks_endpoint():
    response = client.get("/api/v1/lti/jwks")
    assert response.status_code == 200
    data = response.json()
    assert "keys" in data
    assert len(data["keys"]) > 0
    key = data["keys"][0]
    assert key["kty"] == "RSA"
    assert key["alg"] == "RS256"

def test_login_init_endpoint():
    response = client.get("/api/v1/lti/login?iss=test&login_hint=hint", follow_redirects=False)
    # FastAPI test client follows redirects by default if not set, or returns 307
    assert response.status_code in (302, 307, 303)
    
def test_launch_invalid_state():
    response = client.post("/api/v1/lti/launch", data={"state": "invalid", "id_token": "fake"})
    assert response.status_code == 400
    assert "Invalid state" in response.text
