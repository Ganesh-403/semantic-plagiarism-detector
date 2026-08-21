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

import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

logger = logging.getLogger(__name__)

# ─── Repository root resolution ────────────────────────────────────────────
# All paths are anchored to the repository root (the directory that contains
# ``src/``, ``app/``, ``tests/``, etc.).  Resolving once at import time keeps
# behavior deterministic and immune to the current working directory of the
# process that imports this module.
_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

# ─── Application display config (pre-existing) ─────────────────────────────
DEFAULT_APP_TITLE: Final[str] = "Semantic Plagiarism Detection System"
DEFAULT_PDF_FOOTER_TEXT: Final[str] = ""

DEFAULT_VALID_ROLES: Final[set[str]] = {"admin", "teacher"}


def get_valid_roles() -> set[str]:
    """Return the set of valid user roles.

    Configured via the ``ALLOWED_USER_ROLES`` environment variable as a
    comma-separated string (e.g. ``ALLOWED_USER_ROLES="admin,teacher,teaching_assistant"``).
    If not set or empty, falls back to ``{"admin", "teacher"}``.
    Role names are normalized to lowercase and stripped of leading/trailing whitespace.
    """
    raw = os.getenv("ALLOWED_USER_ROLES", "").strip()
    if not raw:
        return set(DEFAULT_VALID_ROLES)
    roles = {role.strip().lower() for role in raw.split(",") if role.strip()}
    return roles or set(DEFAULT_VALID_ROLES)


SUPPORTED_OCR_LANGUAGES = {
    "eng": "English",
    "spa": "Spanish",
    "fra": "French",
    "deu": "German",
    "por": "Portuguese",
    "ita": "Italian",
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
FALLBACK_DATA_DIR: Final[Path] = (
    Path(tempfile.gettempdir()) / "semantic_plagiarism_detector" / "data"
)
FALLBACK_CORPUS_DB_PATH: Final[Path] = FALLBACK_DATA_DIR / "corpus.db"

# Backup directory configuration (issue #2790).
# Loads BACKUP_DIR environment variable, defaulting to an absolute path
# completely outside the repository/web root (e.g., /var/backups/spd/).
_DEFAULT_BACKUP_DIR_STR: Final[str] = "/var/backups/spd"


def get_backup_dir() -> Path:
    """Return the resolved absolute Path for storing database backups.

    Configured via the ``BACKUP_DIR`` environment variable.
    Ensures default location is an absolute path completely outside the repository/web root (Issue #2790).
    """
    raw_val = os.getenv("BACKUP_DIR", "").strip()
    if raw_val:
        return Path(raw_val).resolve()
    default_path = Path(_DEFAULT_BACKUP_DIR_STR)
    if not default_path.is_absolute():
        default_path = default_path.resolve()
    return default_path


BACKUP_DIR: Final[Path] = get_backup_dir()


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


def get_api_support_contact() -> dict[str, str]:
    """Return the OpenAPI `contact` object, driven by configuration.

    Different universities deploying their own instance should show
    their own local IT support contact in the generated API docs rather
    than a hardcoded placeholder. Reads ``API_SUPPORT_EMAIL`` and
    ``API_SUPPORT_URL`` from the environment; each key is included in
    the returned dict only when its corresponding environment variable
    is set to a non-blank value, matching the OpenAPI Contact Object's
    spec where ``name``, ``url``, and ``email`` are all optional. If
    neither is configured, only ``name`` is returned.
    """
    contact_info: dict[str, str] = {"name": "API Support"}

    support_url = os.getenv("API_SUPPORT_URL", "").strip()
    if support_url:
        contact_info["url"] = support_url

    support_email = os.getenv("API_SUPPORT_EMAIL", "").strip()
    if support_email:
        contact_info["email"] = support_email

    return contact_info


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
        if timeout_minutes < 1:
            logger.warning(
                "Invalid backup timeout %d, defaulting to 30",
                timeout_minutes,
            )
            return 30 * 60
        return timeout_minutes * 60
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


# ── Branding Configuration (Issue #2025) ──────────────────────────────────────


@dataclass
class BrandingConfig:
    """Dataclass representing the white-label branding configuration.

    This structure is used by the Streamlit dashboard and PDF report generator
    to apply custom logos, colors, and text to the application interface.

    Attributes:
        app_name: The display name of the application.
        tagline: A short subtitle or description shown below the app name.
        primary_color: Hex color code for primary UI elements and buttons.
        secondary_color: Hex color code for secondary accents and headers.
        logo_path: Relative or absolute path to the logo image file.
        footer_text: Copyright or attribution text displayed in the footer.
    """

    app_name: str = "Semantic Plagiarism Detector"
    tagline: str = "Advanced AI-Powered Academic Integrity Tool"
    primary_color: str = "#2563EB"
    secondary_color: str = "#1E40AF"
    logo_path: str = "assets/logo.png"
    footer_text: str = "© 2024 Semantic Plagiarism Detector. All rights reserved."

    def to_dict(self) -> dict:
        """Convert the dataclass to a dictionary for JSON serialization."""
        return asdict(self)


def load_branding_config(config_path: Path | str | None = None) -> BrandingConfig:
    """Load and validate branding configuration from a JSON file.

    Reads the specified JSON file and maps its contents to a `BrandingConfig`
    dataclass. If the file is missing, unreadable, or contains invalid JSON,
    the function gracefully falls back to the default configuration defined
    in the `BrandingConfig` dataclass.

    Args:
        config_path: Path to the branding_config.json file. If None, defaults
                     to `config/branding_config.json` relative to the repo root.

    Returns:
        A validated `BrandingConfig` instance.

    Examples:
        >>> config = load_branding_config()
        >>> print(config.app_name)
        'Semantic Plagiarism Detector'
    """
    # Determine default path if none provided
    if config_path is None:
        # Assume this file is in src/core/, so repo root is parents[2]
        repo_root = Path(__file__).resolve().parents[2]
        config_path = repo_root / "config" / "branding_config.json"
    else:
        config_path = Path(config_path)

    # Initialize with defaults
    config = BrandingConfig()

    if not config_path.exists():
        logger.warning(
            "Branding config file not found at %s. Using default values.",
            config_path,
        )
        return config

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            logger.error(
                "Branding config at %s is not a JSON object. Using defaults.",
                config_path,
            )
            return config

        # Update config with values from JSON, ignoring unknown keys
        for key, value in data.items():
            if hasattr(config, key):
                # Basic type validation
                expected_type = type(getattr(config, key))
                if isinstance(value, expected_type):
                    setattr(config, key, value)
                else:
                    logger.warning(
                        "Invalid type for branding config '%s': expected %s, got %s. Ignoring.",
                        key,
                        expected_type.__name__,
                        type(value).__name__,
                    )
            else:
                logger.debug("Ignoring unknown branding config key: %s", key)

        logger.info("Successfully loaded branding config from %s", config_path)
        return config

    except json.JSONDecodeError as exc:
        logger.error(
            "Failed to parse branding config JSON at %s: %s. Using defaults.",
            config_path,
            exc,
        )
        return BrandingConfig()
    except Exception as exc:
        logger.error(
            "Unexpected error reading branding config at %s: %s. Using defaults.",
            config_path,
            exc,
        )
        return BrandingConfig()


# Global cached instance to avoid repeated file I/O
_BRANDING_CONFIG_CACHE: BrandingConfig | None = None


def get_branding_config() -> BrandingConfig:
    """Return the cached branding configuration.

    Loads the configuration from disk on the first call and caches it
    in memory for subsequent calls. This prevents repeated file I/O
    during Streamlit reruns.

    Returns:
        The global `BrandingConfig` instance.
    """
    global _BRANDING_CONFIG_CACHE
    if _BRANDING_CONFIG_CACHE is None:
        _BRANDING_CONFIG_CACHE = load_branding_config()
    return _BRANDING_CONFIG_CACHE


def clear_branding_config_cache() -> None:
    """Clear the cached branding configuration.

    Useful for testing or when the configuration file is modified at runtime.
    """
    global _BRANDING_CONFIG_CACHE
    _BRANDING_CONFIG_CACHE = None


def get_rescan_interval_minutes() -> int:
    """Return the configured corpus rescan interval in minutes (default: 0 / disabled)."""
    val = os.getenv("CORPUS_RESCAN_INTERVAL_MINUTES", "0")
    try:
        return max(0, int(val))
    except (ValueError, TypeError):
        return 0


def __getattr__(name: str):
    if name == "VALID_ROLES":
        return get_valid_roles()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
