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

"""
tests/test_dockerfile_healthcheck.py
--------------------------------------
Guards the Dockerfile's HEALTHCHECK instruction so orchestrators (Docker
Compose, Kubernetes) can tell when the Streamlit app is actually ready to
serve traffic.

Targets Streamlit's built-in /_stcore/health endpoint on port 8501 rather
than the app's /healthz route: /healthz is served by a separate FastAPI
process on port 8000 (see app/streamlit_app.py's background uvicorn
thread), which this image doesn't publish, so curling it from inside the
container at localhost:8501 would never succeed. /_stcore/health is also
what docker-compose.yml's "app" service healthcheck already relies on,
so this keeps both healthcheck definitions consistent.
"""

from __future__ import annotations

import re
from pathlib import Path

DOCKERFILE_PATH = Path("Dockerfile")


def _dockerfile_text() -> str:
    return DOCKERFILE_PATH.read_text(encoding="utf-8")


def test_dockerfile_has_a_healthcheck_instruction():
    text = _dockerfile_text()
    assert "HEALTHCHECK" in text


def test_healthcheck_uses_required_timing_flags():
    text = _dockerfile_text()
    healthcheck_line = next(
        line for line in text.splitlines() if line.strip().startswith("HEALTHCHECK")
    )
    assert "--interval=30s" in healthcheck_line
    assert "--timeout=10s" in healthcheck_line
    assert "--retries=3" in healthcheck_line


def test_healthcheck_curls_a_reachable_port_8501_endpoint():
    """The HEALTHCHECK must target a route actually served on port 8501
    inside the container (Streamlit's own listener), not the separate
    background FastAPI process on port 8000."""
    text = _dockerfile_text()
    match = re.search(r"CMD\s+curl\s+--fail\s+(\S+)", text)
    assert match is not None, "HEALTHCHECK CMD with curl --fail not found"

    url = match.group(1)
    assert "localhost:8501" in url or "127.0.0.1:8501" in url
    assert ":8000" not in url  # background FastAPI server's port, not published


def test_healthcheck_fails_over_with_exit_1():
    text = _dockerfile_text()
    healthcheck_block = text[text.index("HEALTHCHECK") :]
    healthcheck_block = healthcheck_block[: healthcheck_block.index("\n\n")]
    assert "|| exit 1" in healthcheck_block


def test_curl_is_installed_in_the_image():
    """The HEALTHCHECK CMD depends on curl being present in the final image."""
    text = _dockerfile_text()
    assert re.search(r"^\s*curl\s*\\?\s*$", text, re.MULTILINE) is not None
