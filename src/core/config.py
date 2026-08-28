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
config.py
---------
Configuration management and validation for branding settings.
Provides safe loading and validation of branding_config.json with graceful fallbacks.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, Optional

# Set up logging
logger = logging.getLogger(__name__)

# Default branding configuration
DEFAULT_BRAND_COLOR = "#1e3a8a"
DEFAULT_LOGO_PATH = None

# Path to branding config file (relative to project root)
BRANDING_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "config",
    "branding_config.json",
)


def validate_hex_color(color: str) -> bool:
    """
    Validate that a string is a valid hex color code.

    Args:
        color: Color string to validate

    Returns:
        True if valid hex color (#RRGGBB or #RGB), False otherwise
    """
    if not color or not isinstance(color, str):
        return False

    # Match #RRGGBB or #RGB format
    pattern = re.compile(r"^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$")
    return bool(pattern.match(color))


class BrandingConfig:
    """
    validated branding configuration with safe defaults.
    """

    def __init__(
        self,
        brand_color: str = DEFAULT_BRAND_COLOR,
        logo_path: Optional[str] = DEFAULT_LOGO_PATH,
    ):
        """
        Initialize branding configuration.

        Args:
            brand_color: Hex color code for branding
            logo_path: Optional path to logo file
        """
        self.brand_color = brand_color
        self.logo_path = logo_path

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BrandingConfig":
        """
        Create BrandingConfig from dictionary with validation.

        Args:
            data: Dictionary containing branding configuration

        Returns:
            BrandingConfig instance with validated values
        """
        brand_color = data.get("brand_color", DEFAULT_BRAND_COLOR)
        logo_path = data.get("logo_path", DEFAULT_LOGO_PATH)

        # Validate brand_color
        if brand_color is not None and not validate_hex_color(brand_color):
            logger.warning(
                f"Invalid brand_color format: '{brand_color}'. "
                f"Expected hex color (#RRGGBB or #RGB). Using default: {DEFAULT_BRAND_COLOR}"
            )
            brand_color = DEFAULT_BRAND_COLOR

        # Validate logo_path (allow None or string)
        if logo_path is not None and not isinstance(logo_path, str):
            logger.warning(
                f"Invalid logo_path type: {type(logo_path).__name__}. "
                f"Expected string or None. Using default: {DEFAULT_LOGO_PATH}"
            )
            logo_path = DEFAULT_LOGO_PATH

        return cls(brand_color=brand_color, logo_path=logo_path)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert BrandingConfig to dictionary.

        Returns:
            Dictionary representation of the configuration
        """
        return {
            "brand_color": self.brand_color,
            "logo_path": self.logo_path,
        }


def load_branding_config(config_path: Optional[str] = None) -> BrandingConfig:
    """
    Load and validate branding configuration from JSON file.

    Args:
        config_path: Optional path to branding config file.
                    If not provided, uses default BRANDING_CONFIG_PATH.

    Returns:
        BrandingConfig instance with validated values or defaults if loading fails
    """
    if config_path is None:
        config_path = BRANDING_CONFIG_PATH

    # Check if file exists
    if not os.path.exists(config_path):
        logger.info(
            f"Branding config file not found at {config_path}. Using default branding configuration."
        )
        return BrandingConfig()

    # Try to load and parse JSON
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        logger.warning(
            f"Invalid JSON in branding config file {config_path}: {e}. Using default branding configuration."
        )
        return BrandingConfig()
    except Exception as e:
        logger.warning(
            f"Error reading branding config file {config_path}: {e}. Using default branding configuration."
        )
        return BrandingConfig()

    # Validate and create config
    try:
        config = BrandingConfig.from_dict(data)
        logger.info(f"Successfully loaded branding configuration from {config_path}")
        return config
    except Exception as e:
        logger.warning(
            f"Error validating branding configuration: {e}. Using default branding configuration."
        )
        return BrandingConfig()


# Global branding configuration instance
_branding_config: Optional[BrandingConfig] = None


def get_branding_config() -> BrandingConfig:
    """
    Get the global branding configuration instance.
    Loads on first call and caches the result.

    Returns:
        BrandingConfig instance
    """
    global _branding_config
    if _branding_config is None:
        _branding_config = load_branding_config()
    return _branding_config


def reload_branding_config() -> BrandingConfig:
    """
    Force reload the branding configuration from file.
    Useful for testing or when config file changes at runtime.

    Returns:
        Newly loaded BrandingConfig instance
    """
    global _branding_config
    _branding_config = load_branding_config()
    return _branding_config


"""Central plagiarism threshold and severity configuration."""

import os
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Final, Mapping

LOW_SEVERITY: Final[str] = "Low"
MEDIUM_SEVERITY: Final[str] = "Medium"
HIGH_SEVERITY: Final[str] = "High"
CRITICAL_SEVERITY: Final[str] = "Critical"

SEVERITY_ORDER: Final[tuple[str, ...]] = (
    LOW_SEVERITY,
    MEDIUM_SEVERITY,
    HIGH_SEVERITY,
    CRITICAL_SEVERITY,
)
SEVERITY_RANK: Final[Mapping[str, int]] = {
    label: rank for rank, label in enumerate(SEVERITY_ORDER)
}

# Embedding batch size configuration (default: 32)
EMBEDDING_BATCH_SIZE: Final[int] = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))

# Minimum consecutive words for a highlighted match in side-by-side diffs.
DEFAULT_DIFF_MIN_MATCH_LENGTH: Final[int] = int(
    os.getenv("DEFAULT_DIFF_MIN_MATCH_LENGTH", "4")
)


@dataclass(frozen=True)
class SimilarityThresholds:
    """Validated plagiarism and severity boundaries."""

    plagiarism: float = 0.59
    medium: float = 0.75
    high: float = 0.90

    def __post_init__(self) -> None:
        plagiarism = _validate_boundary("plagiarism", self.plagiarism)
        medium = _validate_boundary("medium", self.medium)
        high = _validate_boundary("high", self.high)

        if not plagiarism <= medium <= high:
            raise ValueError(
                "Thresholds must satisfy " "0.0 <= plagiarism <= medium <= high <= 1.0."
            )

        object.__setattr__(self, "plagiarism", plagiarism)
        object.__setattr__(self, "medium", medium)
        object.__setattr__(self, "high", high)


def _validate_boundary(name: str, value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} threshold must be a real number.")

    numeric = float(value)
    if not isfinite(numeric):
        raise ValueError(f"{name} threshold must be finite.")
    if not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{name} threshold must be between 0.0 and 1.0.")
    return numeric


def validate_thresholds(
    thresholds: SimilarityThresholds,
) -> SimilarityThresholds:
    """Validate threshold type and ordering."""
    if not isinstance(thresholds, SimilarityThresholds):
        raise TypeError("thresholds must be a SimilarityThresholds instance.")

    if not thresholds.plagiarism <= thresholds.medium <= thresholds.high:
        raise ValueError(
            "Thresholds must satisfy " "0.0 <= plagiarism <= medium <= high <= 1.0."
        )

    return thresholds


DEFAULT_THRESHOLDS: Final[SimilarityThresholds] = SimilarityThresholds()
PLAGIARISM_THRESHOLD: Final[float] = DEFAULT_THRESHOLDS.plagiarism

# Default location of an optional on-disk threshold config (Issue #2267).
# When present, load_threshold_config() reads recommended boundaries from it;
# when absent (the default), behavior is unchanged and DEFAULT_THRESHOLDS is used.
THRESHOLD_CONFIG_PATH: Final[str] = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "config",
    "thresholds.json",
)


def load_threshold_config(
    config_path: Optional[str] = None,
) -> SimilarityThresholds:
    """Load recommended similarity thresholds from a JSON config file.

    Reads a JSON object with optional ``plagiarism``, ``medium`` and ``high``
    keys (each in the inclusive ``[0.0, 1.0]`` range) and returns a validated
    :class:`SimilarityThresholds` instance.

    When *config_path* is ``None`` the environment variable
    ``THRESHOLD_CONFIG_PATH`` (falling back to ``config/thresholds.json``
    relative to the repo root) is used.

    If the file is missing, unreadable, or contains invalid values the
    defaults are returned unchanged, so supplying no calibration config never
    alters existing detection behavior.

    Args:
        config_path: Optional explicit path to the threshold JSON file.

    Returns:
        The validated thresholds from the file, or ``DEFAULT_THRESHOLDS``.
    """
    if config_path is None:
        config_path = os.getenv("THRESHOLD_CONFIG_PATH", THRESHOLD_CONFIG_PATH)

    if not os.path.exists(config_path):
        return DEFAULT_THRESHOLDS

    try:
        with open(config_path, "r", encoding="utf-8") as stream:
            data = json.load(stream)
        if not isinstance(data, dict):
            logger.warning(
                "Threshold config %s is not a JSON object. Using defaults.",
                config_path,
            )
            return DEFAULT_THRESHOLDS
        return SimilarityThresholds(
            plagiarism=data.get("plagiarism", DEFAULT_THRESHOLDS.plagiarism),
            medium=data.get("medium", DEFAULT_THRESHOLDS.medium),
            high=data.get("high", DEFAULT_THRESHOLDS.high),
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.warning(
            "Failed to load threshold config from %s: %s. Using defaults.",
            config_path,
            exc,
        )
        return DEFAULT_THRESHOLDS


def normalize_score(score: Real) -> float:
    """Return a finite score clamped to the inclusive [0, 1] range."""
    if isinstance(score, bool) or not isinstance(score, Real):
        raise TypeError("Similarity score must be a real number.")

    value = float(score)
    if not isfinite(value):
        raise ValueError("Similarity score must be finite.")
    return min(1.0, max(0.0, value))


def is_plagiarism(
    score: Real,
    threshold: Real = DEFAULT_THRESHOLDS.plagiarism,
) -> bool:
    """Return whether a score reaches a validated flagging threshold."""
    normalized_score = normalize_score(score)
    normalized_threshold = _validate_boundary("plagiarism", threshold)
    return normalized_score >= normalized_threshold


def severity_from_score(
    score: Real,
    thresholds: SimilarityThresholds = DEFAULT_THRESHOLDS,
) -> str:
    """Return the canonical Low, Medium, or High severity label."""
    validate_thresholds(thresholds)
    normalized = normalize_score(score)

    if normalized < thresholds.plagiarism:
        return LOW_SEVERITY
    if normalized >= thresholds.high:
        return HIGH_SEVERITY
    if normalized >= thresholds.medium:
        return MEDIUM_SEVERITY
    return LOW_SEVERITY


def severity_key(
    score: Real,
    thresholds: SimilarityThresholds = DEFAULT_THRESHOLDS,
) -> str:
    """Return a lowercase key for colors and presentation helpers."""
    return severity_from_score(score, thresholds).lower()


def normalize_severity_label(label: str) -> str:
    """Normalize supported canonical and legacy severity labels."""
    clean = str(label or "").strip().lower()

    severity_aliases = {
        "low": LOW_SEVERITY,
        "medium": MEDIUM_SEVERITY,
        "high": HIGH_SEVERITY,
        "critical": CRITICAL_SEVERITY,
        "🟢 low": LOW_SEVERITY,
        "🟡 medium": MEDIUM_SEVERITY,
        "🔴 high": HIGH_SEVERITY,
        "ðÿ”´ high": HIGH_SEVERITY,
        "warning": MEDIUM_SEVERITY,
    }

    try:
        return severity_aliases[clean]
    except KeyError:
        raise ValueError(f"Unknown severity label: {label!r}") from None


def severity_rank(label: str) -> int:
    """Return a stable sort rank for a severity label."""
    return SEVERITY_RANK[normalize_severity_label(label)]


# ============================================================================
# OFFLINE MODE CONFIGURATION
# ============================================================================


def get_offline_mode_status() -> bool:
    """Check if offline mode is enabled."""
    import os

    return os.getenv("OFFLINE_MODE", "false").lower() == "true"


def get_offline_config() -> dict[str, Any]:
    """Get offline mode configuration."""
    import os

    return {
        "enabled": get_offline_mode_status(),
        "cache_dir": os.getenv("OFFLINE_CACHE_DIR", ".cache/offline"),
        "model_cache_dir": os.getenv("OFFLINE_MODEL_CACHE_DIR", ".cache/models"),
        "max_cache_size_mb": int(os.getenv("OFFLINE_MAX_CACHE_SIZE_MB", "500")),
        "preload_models": os.getenv("OFFLINE_PRELOAD_MODELS", "true").lower() == "true",
        "disable_telemetry": os.getenv("OFFLINE_DISABLE_TELEMETRY", "true").lower()
        == "true",
    }


def test_branding_config_path_exists():
    """Test that BRANDING_CONFIG_PATH resolves to an existing file."""
    config_path = config_module.BRANDING_CONFIG_PATH  # noqa: F821

    assert os.path.isfile(config_path)
