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


def test_rate_limit_endpoint():
    response = client.get(
        "/api/v1/rate_limit",
        headers={"Authorization": "Bearer dummy-token"},
    )

    assert response.status_code == 200

    data = response.json()

    assert "limit" in data
    assert "remaining" in data
    assert "reset_in_seconds" in data

    initial_remaining = data["remaining"]
    assert initial_remaining <= data["limit"]

    # Consume one token
    from src.security.rate_limiter import get_token_bucket_limiter

    lim = get_token_bucket_limiter()
    lim.consume("dummy-token")

    # Fetch rate limit status again
    response = client.get(
        "/api/v1/rate_limit",
        headers={"Authorization": "Bearer dummy-token"},
    )
    assert response.status_code == 200
    data2 = response.json()
    assert data2["remaining"] == initial_remaining - 1
