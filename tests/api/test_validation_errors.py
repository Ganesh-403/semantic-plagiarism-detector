# MIT License
#
# Copyright (c) 2026 Ganesh Kambli
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from fastapi.testclient import TestClient

from src.api.app import app

client = TestClient(app)


def test_validation_error_format():
    response = client.post(
        "/api/v1/scan",
        headers={
            "Authorization": "Bearer dummy-token",
            "Content-Type": "multipart/form-data; boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW",
        },
    )

    assert response.status_code == 422

    data = response.json()

    assert data["error"] is True
    assert "message" in data
    assert "details" in data
    assert isinstance(data["details"], list)


def test_value_error_returns_400_bad_request():
    """Verify that a ValueError in request handling returns HTTP 400 Bad Request."""
    from fastapi import APIRouter

    from src.api.app import app

    test_router = APIRouter()

    @test_router.get("/api/v1/test-value-error")
    def trigger_value_error(username: str = ""):
        from src.db.auth import _validate_username

        _validate_username(username)
        return {"status": "ok"}

    app.include_router(test_router)

    response = client.get(
        "/api/v1/test-value-error?username=",
        headers={"Authorization": "Bearer dummy-token"},
    )
    assert response.status_code == 400
    data = response.json()
    assert data["error"] is True
    assert data["code"] == 400
    assert "Username cannot be empty" in data["message"]
