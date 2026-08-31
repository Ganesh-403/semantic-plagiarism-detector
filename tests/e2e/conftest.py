"""
tests/e2e/conftest.py
---------------------
Shared fixtures for the Playwright E2E suite (Issue #3030).

Responsibilities
~~~~~~~~~~~~~~~~
- Launch a real Streamlit server against an isolated, temporary SQLite
  auth database so the E2E test never touches the developer's local
  ``users.db`` / ``corpus.db``.
- Seed a deterministic test user (``e2e_user`` / ``TestPass123!``) with
  the ``teacher`` role — matching the role expected by the non-admin
  "Secure Student Search Portal" flow in ``app/streamlit_app.py``.
- Provide a synchronous ``page`` fixture so individual test files stay
  short and focused on the critical path.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterator

import pytest
from playwright.sync_api import Page, sync_playwright

REPO_ROOT = Path(__file__).resolve().parents[2]

TEST_USERNAME = "e2e_user"
TEST_PASSWORD = "TestPass123!"
TEST_ROLE = "teacher"


def _pick_free_port() -> int:
    """Return an OS-assigned free TCP port for the Streamlit server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_streamlit(url: str, timeout_s: float = 90.0) -> None:
    """Poll the Streamlit health endpoint until the server is ready."""
    import urllib.error
    import urllib.request

    deadline = time.time() + timeout_s
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/_stcore/health", timeout=2) as r:
                if r.status == 200 and r.read().strip() == b"ok":
                    return
        except Exception as exc:
            last_err = exc
            time.sleep(0.5)
    raise RuntimeError(
        f"Streamlit server at {url} did not become healthy in {timeout_s}s: {last_err}"
    )


def _seed_test_user(auth_db_path: Path) -> None:
    """Create the test user directly in the isolated auth DB.

    Done via the project's own ``add_user`` helper so the password is
    hashed with the same Argon2 parameters the live app uses.
    """
    os.environ["AUTH_DB_PATH"] = str(auth_db_path)
    os.environ["CORPUS_DB_PATH"] = str(auth_db_path.parent / "e2e_corpus.db")

    import importlib

    from src.core import app_config

    importlib.reload(app_config)
    from src.db import auth as auth_mod

    importlib.reload(auth_mod)

    auth_mod.init_db()
    try:
        auth_mod.add_user(TEST_USERNAME, TEST_PASSWORD, role=TEST_ROLE)
    except ValueError:
        # User already exists from a previous run in the same temp DB.
        pass


@pytest.fixture(scope="session")
def streamlit_url(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """Launch Streamlit against an isolated DB and yield its base URL."""
    work_dir = tmp_path_factory.mktemp("e2e_streamlit")
    auth_db = work_dir / "users.db"

    _seed_test_user(auth_db)

    port = _pick_free_port()
    url = f"http://127.0.0.1:{port}"

    env = os.environ.copy()
    env.update(
        {
            "MPLBACKEND": "Agg",
            "STREAMLIT_SERVER_HEADLESS": "true",
            "STREAMLIT_SERVER_FILEWATCHER_TYPE": "none",
            "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false",
            "STREAMLIT_BROWSER_SERVER_ADDRESS": "127.0.0.1",
            "AUTH_DB_PATH": str(auth_db),
            "CORPUS_DB_PATH": str(work_dir / "e2e_corpus.db"),
        }
    )

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(REPO_ROOT / "app" / "streamlit_app.py"),
            "--server.port",
            str(port),
            "--server.address",
            "127.0.0.1",
            "--server.headless",
            "true",
            "--browser.gatherUsageStats",
            "false",
            "--global.developmentMode",
            "false",
        ],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        _wait_for_streamlit(url)
        yield url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


@pytest.fixture
def page(streamlit_url: str) -> Iterator[Page]:
    """Yield a fresh browser page navigated to the Streamlit app."""
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1280, "height": 900},
            locale="en-US",
        )
        pg = context.new_page()
        pg.set_default_timeout(15_000)
        pg.goto(streamlit_url, wait_until="domcontentloaded")
        yield pg
        context.close()
        browser.close()


@pytest.fixture
def authenticated_page(page: Page) -> Page:
    """A page that has completed the login step."""
    from tests.e2e.pages.login_page import LoginPage

    login = LoginPage(page)
    login.login(TEST_USERNAME, TEST_PASSWORD)
    return page
