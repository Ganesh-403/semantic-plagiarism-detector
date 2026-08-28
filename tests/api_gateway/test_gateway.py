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

"""Unit tests for Core APIGateway routing, auth, rate limiting, and dispatch."""

import asyncio

from src.api_gateway.gateway import (
    APIGateway,
    InvalidAPIKeyException,
    RateLimitExceededException,
    RouteNotFoundException,
)


def create_gateway():
    gw = APIGateway()
    gw.rate_limiter.default_limit = 2
    return gw


def sample_handler(doc_id: str = "doc_1") -> dict:
    return {"status": "ok", "doc_id": doc_id}


def test_register_and_dispatch_route():
    async def run():
        gateway = create_gateway()
        gateway.register(
            "POST", "/documents/check", sample_handler, require_api_key=False
        )

        res = await gateway.dispatch("POST", "/documents/check", doc_id="doc_42")
        assert res == {"status": "ok", "doc_id": "doc_42"}

    asyncio.run(run())


def test_authenticated_request_succeeds():
    async def run():
        gateway = create_gateway()
        gateway.register("GET", "/secure/data", sample_handler, require_api_key=True)

        _, raw_key = gateway.api_keys.create_key("test-key")

        res = await gateway.dispatch("GET", "/secure/data", api_key=raw_key)
        assert res == {"status": "ok", "doc_id": "doc_1"}

    asyncio.run(run())


def test_missing_and_invalid_api_key_rejected():
    async def run():
        gateway = create_gateway()
        gateway.register("GET", "/protected", sample_handler, require_api_key=True)

        # Missing API key
        try:
            await gateway.dispatch("GET", "/protected", api_key=None)
            assert False, "Should have raised InvalidAPIKeyException"
        except InvalidAPIKeyException as exc:
            assert "required" in str(exc)

        # Invalid API key
        try:
            await gateway.dispatch("GET", "/protected", api_key="invalid_key")
            assert False, "Should have raised InvalidAPIKeyException"
        except InvalidAPIKeyException as exc:
            assert "Invalid" in str(exc)

    asyncio.run(run())


def test_rate_limit_exceeded_raises_exception():
    async def run():
        gateway = create_gateway()
        gateway.register("GET", "/limited", sample_handler, require_api_key=False)

        await gateway.dispatch("GET", "/limited", client_ip="1.2.3.4")
        await gateway.dispatch("GET", "/limited", client_ip="1.2.3.4")

        try:
            await gateway.dispatch("GET", "/limited", client_ip="1.2.3.4")
            assert False, "Should have raised RateLimitExceededException"
        except RateLimitExceededException as exc:
            assert "Rate limit exceeded" in str(exc)

    asyncio.run(run())


def test_unregistered_route_raises_not_found():
    async def run():
        gateway = create_gateway()
        try:
            await gateway.dispatch("GET", "/unknown/route")
            assert False, "Should have raised RouteNotFoundException"
        except RouteNotFoundException as exc:
            assert "No route registered" in str(exc)

    asyncio.run(run())
