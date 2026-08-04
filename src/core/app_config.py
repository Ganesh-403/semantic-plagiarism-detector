"""Application-level environment configuration.

This module is the single source of truth for paths and tunables that
previously were re-derived independently in many modules.  Importing from
here guarantees every caller resolves the same on-disk location for a given
resource, eliminating the historical drift in which ``corpus.db`` was
computed three different ways and ``corpus.index`` four different ways.

Centralized here (issue #618):

* ``CORPUS_DB_PATH``         – shared SQLite DB for documents / chunks /
                                embeddings / incidents / translation cache.
* ``AUTH_DB_PATH``           – SQLite DB for user authentication.
* ``FAISS_INDEX_PATH``       – on-disk FAISS index file.
* ``HEALTHZ_DB_PATHS``       – tuple of DB files whose sizes are summed by
                                the ``/healthz`` endpoint.
* ``FALLBACK_CORPUS_DB_PATH`` – temp-dir fallback used when the primary
                                corpus DB directory is not writable.
* ``FALLBACK_DATA_DIR``      – parent dir of ``FALLBACK_CORPUS_DB_PATH``.

Existing per-module mutators (``src.db.corpus_db.configure_db_path``,
``src.db.auth.configure_db_path``) and per-module path constants
(``_DB_PATH`` / ``DEFAULT_DB_PATH`` / ``DB_PATH``) are preserved as thin
wrappers seeded from these constants so that tests which import or monkey-patch
the legacy names continue to work without modification.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Final

# ─── Repository root resolution ────────────────────────────────────────────
# All paths are anchored to the repository root (the directory that contains
# ``src/``, ``app/``, ``tests/``, etc.).  Resolving once at import time keeps
# behavior deterministic and immune to the current working directory of the
# process that imports this module.
_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

# ─── Application display config (pre-existing) ─────────────────────────────
DEFAULT_APP_TITLE: Final[str] = "Semantic Plagiarism Detection System"
DEFAULT_PDF_FOOTER_TEXT: Final[str] = ""

SUPPORTED_OCR_LANGUAGES = {
    "eng": "English",
    "spa": "Spanish",
    "fra": "French",
}


# ─── Database / index path constants (issue #618) ──────────────────────────

# Primary corpus DB.  Lives under ``<repo>/data/corpus.db`` – this matches the
# original resolution used by ``src/db/corpus_db.py`` (and incidentally by
# ``src/db/incidents.py`` and ``src/db/translation_cache.py``).
CORPUS_DB_PATH: Final[Path] = _REPO_ROOT / "data" / "corpus.db"

# Auth DB.  Lives directly at the repo root (``<repo>/users.db``) – this
# matches the original resolution used by ``src/db/auth.py``.
AUTH_DB_PATH: Final[Path] = _REPO_ROOT / "users.db"

# FAISS index file.  Lives directly at the repo root (``<repo>/corpus.index``)
# – this matches the original resolution used by ``app/streamlit_app.py``,
# ``src/api/app.py`` and ``src/utils/mock_data.py``.  (``src/cli.py``
# previously resolved it to ``<repo>/src/corpus.index``, which was a latent
# bug; centralizing it here fixes that drift.)
FAISS_INDEX_PATH: Final[Path] = _REPO_ROOT / "corpus.index"

# DB files inspected by the ``/healthz`` endpoint.  The two real SQLite DBs
# are the corpus DB and the auth DB.  (The original implementation in
# ``src/api/app.py`` inspected ``<repo>/corpus.db`` instead of
# ``<repo>/data/corpus.db`` – a latent bug, since no code writes to that
# location.  Centralizing here fixes that drift.)
HEALTHZ_DB_PATHS: Final[tuple[Path, ...]] = (CORPUS_DB_PATH, AUTH_DB_PATH)

# Temp-dir fallback used when the primary data directory is not writable.
# All three of corpus_db.py / incidents.py / translation_cache.py previously
# hard-coded this exact path; it is now centralized here.
FALLBACK_DATA_DIR: Final[Path] = Path(tempfile.gettempdir()) / "semantic_plagiarism_detector" / "data"
FALLBACK_CORPUS_DB_PATH: Final[Path] = FALLBACK_DATA_DIR / "corpus.db"


# ─── Application display accessors (pre-existing) ──────────────────────────

def get_app_title() -> str:
    """Return the configured application title.

    Empty or whitespace-only values fall back to the default so a
    malformed environment variable cannot leave the browser or page
    title blank.
    """
    configured_title = os.getenv("APP_TITLE", "").strip()
    return configured_title or DEFAULT_APP_TITLE


def get_pdf_footer_text() -> str:
    """Return the configured PDF footer text.

    Empty or whitespace-only values fall back to the default.
    """
    configured_footer = os.getenv("PDF_FOOTER_TEXT", "").strip()
    return configured_footer or DEFAULT_PDF_FOOTER_TEXT


def get_welcome_message() -> str:
    """Return the configured dashboard welcome banner message.

    Empty or whitespace-only values return an empty string, meaning no
    banner is shown by default.
    """
    return os.getenv("APP_WELCOME_MESSAGE", "").strip()


def get_lock_timeout() -> int:
    """Return the configured lock timeout in seconds (default 30)."""
    try:
        timeout = int(os.getenv("LOCK_TIMEOUT_SECONDS", "30"))
        return max(1, timeout)
    except ValueError:
        return 30


def get_backup_idle_timeout() -> int:
    """Return the configured backup idle timeout in seconds (default 30 minutes)."""
    try:
        timeout_minutes = int(os.getenv("BACKUP_IDLE_TIMEOUT_MINUTES", "30"))
        return max(1, timeout_minutes) * 60
    except ValueError:
        return 30 * 60


def get_allowed_webhook_domains() -> list[str]:
    """Return the list of allowed webhook domain hostnames.

    Configured via the ``ALLOWED_WEBHOOK_DOMAINS`` environment variable as a
    comma-separated string (e.g. ``hooks.slack.com, discord.com``). Returns an
    empty list if not set or empty (allowing any domain subject to SSRF checks).
    """
    raw = os.getenv("ALLOWED_WEBHOOK_DOMAINS", "").strip()
    if not raw:
        return []
    return [domain.strip().lower() for domain in raw.split(",") if domain.strip()]
