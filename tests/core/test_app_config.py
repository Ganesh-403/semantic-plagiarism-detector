import json
from pathlib import Path  # noqa: F401
from unittest.mock import mock_open, patch  # noqa: F401

import pytest

from src.core.app_config import (
    DEFAULT_APP_TITLE,
    BrandingConfig,
    _get_env_bool,
    _get_env_int,
    clear_branding_config_cache,
    get_api_support_contact,
    get_app_title,
    get_backup_idle_timeout,
    get_branding_config,
    get_env_bool,
    get_env_int,
    get_lock_timeout,
    load_branding_config,
)
# ---------------------------------------------------------------------------
# Tests for get_app_title
# ---------------------------------------------------------------------------


def test_app_title_uses_default_when_variable_is_missing(
    monkeypatch,
):
    monkeypatch.delenv("APP_TITLE", raising=False)

    assert get_app_title() == DEFAULT_APP_TITLE


def test_app_title_uses_environment_value(monkeypatch):
    monkeypatch.setenv(
        "APP_TITLE",
        "Stanford Plagiarism Detector",
    )

    assert get_app_title() == ("Stanford Plagiarism Detector")


def test_app_title_strips_surrounding_whitespace(monkeypatch):
    monkeypatch.setenv(
        "APP_TITLE",
        "  Campus Integrity Portal  ",
    )

    assert get_app_title() == "Campus Integrity Portal"


def test_blank_app_title_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("APP_TITLE", "   ")

    assert get_app_title() == DEFAULT_APP_TITLE


# ---------------------------------------------------------------------------
# Tests for get_lock_timeout
# ---------------------------------------------------------------------------


def test_get_lock_timeout_default(mocker):
    mocker.patch("os.getenv", return_value="30")

    assert get_lock_timeout() == 30


def test_get_lock_timeout_custom(mocker):
    mocker.patch("os.getenv", return_value="60")

    assert get_lock_timeout() == 60


def test_get_lock_timeout_invalid(mocker):
    mocker.patch("os.getenv", return_value="invalid")

    assert get_lock_timeout() == 30


def test_get_lock_timeout_minimum(mocker):
    mocker.patch("os.getenv", return_value="0")

    assert get_lock_timeout() == 1


# ---------------------------------------------------------------------------
# Tests for BrandingConfig Dataclass (Issue #2025)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_cache():
    """Ensure the branding config cache is cleared before and after each test."""
    clear_branding_config_cache()
    yield
    clear_branding_config_cache()


class TestBrandingConfigDataclass:
    """Test suite for the BrandingConfig dataclass structure."""

    def test_default_values(self):
        """Verify all fields have sensible default values."""
        config = BrandingConfig()

        assert config.app_name == "Semantic Plagiarism Detector"
        assert config.tagline == "Advanced AI-Powered Academic Integrity Tool"
        assert config.primary_color == "#2563EB"
        assert config.secondary_color == "#1E40AF"
        assert config.logo_path == "assets/logo.png"
        assert "©" in config.footer_text

    def test_custom_initialization(self):
        """Verify fields can be overridden during initialization."""
        config = BrandingConfig(app_name="Custom App", primary_color="#FF0000")

        assert config.app_name == "Custom App"
        assert config.primary_color == "#FF0000"
        # Other fields should retain defaults
        assert config.tagline == "Advanced AI-Powered Academic Integrity Tool"

    def test_to_dict_serialization(self):
        """Verify to_dict() returns a valid dictionary representation."""
        config = BrandingConfig()
        data = config.to_dict()

        assert isinstance(data, dict)
        assert data["app_name"] == "Semantic Plagiarism Detector"
        assert "primary_color" in data


# ---------------------------------------------------------------------------
# Tests for load_branding_config
# ---------------------------------------------------------------------------


class TestLoadBrandingConfig:
    """Test suite for the load_branding_config() file loader."""

    def test_loads_valid_json(self, tmp_path):
        """Verify loader correctly parses a valid JSON configuration file."""
        config_file = tmp_path / "branding_config.json"
        custom_data = {
            "app_name": "University Portal",
            "primary_color": "#00FF00",
            "footer_text": "Custom Footer",
        }
        config_file.write_text(json.dumps(custom_data), encoding="utf-8")

        config = load_branding_config(config_file)

        assert config.app_name == "University Portal"
        assert config.primary_color == "#00FF00"
        assert config.footer_text == "Custom Footer"
        # Unspecified fields should retain defaults
        assert config.tagline == "Advanced AI-Powered Academic Integrity Tool"

    def test_fallback_on_missing_file(self, tmp_path):
        """Verify loader returns defaults when the config file does not exist."""
        missing_file = tmp_path / "nonexistent.json"

        config = load_branding_config(missing_file)

        # Should return default config
        assert config.app_name == "Semantic Plagiarism Detector"
        assert config.primary_color == "#2563EB"

    def test_fallback_on_invalid_json(self, tmp_path):
        """Verify loader returns defaults when the file contains malformed JSON."""
        config_file = tmp_path / "branding_config.json"
        config_file.write_text("{ this is not valid json }", encoding="utf-8")

        config = load_branding_config(config_file)

        assert config.app_name == "Semantic Plagiarism Detector"

    def test_fallback_on_non_dict_json(self, tmp_path):
        """Verify loader returns defaults when JSON root is not an object."""
        config_file = tmp_path / "branding_config.json"
        config_file.write_text('["array", "instead", "of", "object"]', encoding="utf-8")

        config = load_branding_config(config_file)

        assert config.app_name == "Semantic Plagiarism Detector"

    def test_ignores_unknown_keys(self, tmp_path):
        """Verify loader ignores JSON keys that don't map to dataclass fields."""
        config_file = tmp_path / "branding_config.json"
        data = {
            "app_name": "Test App",
            "unknown_field": "should be ignored",
            "another_fake": 123,
        }
        config_file.write_text(json.dumps(data), encoding="utf-8")

        config = load_branding_config(config_file)

        assert config.app_name == "Test App"
        assert not hasattr(config, "unknown_field")

    def test_ignores_invalid_types(self, tmp_path):
        """Verify loader ignores fields with incorrect types (e.g., int for string)."""
        config_file = tmp_path / "branding_config.json"
        data = {
            "app_name": 12345,  # Invalid: should be string
            "primary_color": "#FF0000",  # Valid
        }
        config_file.write_text(json.dumps(data), encoding="utf-8")

        config = load_branding_config(config_file)

        # app_name should retain default because 12345 is not a string
        assert config.app_name == "Semantic Plagiarism Detector"
        assert config.primary_color == "#FF0000"

    def test_handles_empty_json_object(self, tmp_path):
        """Verify loader handles an empty JSON object {} gracefully."""
        config_file = tmp_path / "branding_config.json"
        config_file.write_text("{}", encoding="utf-8")

        config = load_branding_config(config_file)

        # All fields should be defaults
        assert config.app_name == "Semantic Plagiarism Detector"
        assert config.primary_color == "#2563EB"


# ---------------------------------------------------------------------------
# Tests for get_branding_config Cache
# ---------------------------------------------------------------------------


class TestGetBrandingConfigCache:
    """Test suite for the get_branding_config() caching mechanism."""

    def test_caches_result(self, tmp_path):
        """Verify get_branding_config() caches the result and doesn't re-read file."""
        config_file = tmp_path / "branding_config.json"
        config_file.write_text('{"app_name": "Cached App"}', encoding="utf-8")

        # Patch the loader to track calls
        with patch(
            "src.core.app_config.load_branding_config", wraps=load_branding_config
        ) as mock_load:
            # First call should hit the loader
            config1 = get_branding_config()
            assert mock_load.call_count == 1

            # Second call should use cache
            config2 = get_branding_config()
            assert mock_load.call_count == 1  # Still 1

            assert config1.app_name == config2.app_name

    def test_clear_cache_forces_reload(self, tmp_path):
        """Verify clear_branding_config_cache() forces a fresh file read."""
        config_file = tmp_path / "branding_config.json"
        config_file.write_text('{"app_name": "App V1"}', encoding="utf-8")

        with patch(
            "src.core.app_config.load_branding_config", wraps=load_branding_config
        ) as mock_load:
            get_branding_config()
            assert mock_load.call_count == 1

            clear_branding_config_cache()

            get_branding_config()
            assert mock_load.call_count == 2


# ---------------------------------------------------------------------------
# Tests for get_api_support_contact
# ---------------------------------------------------------------------------


def test_api_support_contact_omits_url_and_email_when_unset(monkeypatch):
    """Acceptance criteria: falls back to empty/omitted if unset -- only
    'name' should be present when neither env var is configured."""
    monkeypatch.delenv("API_SUPPORT_URL", raising=False)
    monkeypatch.delenv("API_SUPPORT_EMAIL", raising=False)

    contact = get_api_support_contact()

    assert contact == {"name": "API Support"}
    assert "url" not in contact
    assert "email" not in contact


def test_api_support_contact_uses_configured_email(monkeypatch):
    monkeypatch.delenv("API_SUPPORT_URL", raising=False)
    monkeypatch.setenv("API_SUPPORT_EMAIL", "ithelp@stanford.edu")

    contact = get_api_support_contact()

    assert contact["email"] == "ithelp@stanford.edu"
    assert "url" not in contact


def test_api_support_contact_uses_configured_url(monkeypatch):
    monkeypatch.setenv("API_SUPPORT_URL", "https://it.stanford.edu/support")
    monkeypatch.delenv("API_SUPPORT_EMAIL", raising=False)

    contact = get_api_support_contact()

    assert contact["url"] == "https://it.stanford.edu/support"
    assert "email" not in contact


def test_api_support_contact_uses_both_when_configured(monkeypatch):
    monkeypatch.setenv("API_SUPPORT_URL", "https://it.stanford.edu/support")
    monkeypatch.setenv("API_SUPPORT_EMAIL", "ithelp@stanford.edu")

    contact = get_api_support_contact()

    assert contact == {
        "name": "API Support",
        "url": "https://it.stanford.edu/support",
        "email": "ithelp@stanford.edu",
    }


def test_api_support_contact_treats_whitespace_only_as_unset(monkeypatch):
    monkeypatch.setenv("API_SUPPORT_URL", "   ")
    monkeypatch.setenv("API_SUPPORT_EMAIL", "   ")

    contact = get_api_support_contact()

    assert contact == {"name": "API Support"}


def test_api_support_contact_no_longer_hardcodes_example_dot_com(monkeypatch):
    """Regression test: the previous hardcoded placeholder
    ('support@example.com' / 'http://example.com/support') must never be
    returned -- every deployment's contact info must come from its own
    environment configuration, not a shared example.com fallback."""
    monkeypatch.delenv("API_SUPPORT_URL", raising=False)
    monkeypatch.delenv("API_SUPPORT_EMAIL", raising=False)

    contact = get_api_support_contact()

    assert contact.get("email") != "support@example.com"
    assert contact.get("url") != "http://example.com/support"


# ---------------------------------------------------------------------------
# Tests for get_backup_idle_timeout
# ---------------------------------------------------------------------------


def test_get_backup_idle_timeout_default(monkeypatch):
    monkeypatch.delenv("BACKUP_IDLE_TIMEOUT_MINUTES", raising=False)
    assert get_backup_idle_timeout() == 30 * 60


def test_get_backup_idle_timeout_valid(monkeypatch):
    monkeypatch.setenv("BACKUP_IDLE_TIMEOUT_MINUTES", "45")
    assert get_backup_idle_timeout() == 45 * 60


def test_get_backup_idle_timeout_negative_logs_warning_and_defaults_to_30(
    monkeypatch, caplog
):
    monkeypatch.setenv("BACKUP_IDLE_TIMEOUT_MINUTES", "-10")
    with caplog.at_level("WARNING"):
        result = get_backup_idle_timeout()

    assert result == 30 * 60
    assert "Invalid backup timeout -10, defaulting to 30" in caplog.text


def test_get_backup_idle_timeout_zero_logs_warning_and_defaults_to_30(
    monkeypatch, caplog
):
    monkeypatch.setenv("BACKUP_IDLE_TIMEOUT_MINUTES", "0")
    with caplog.at_level("WARNING"):
        result = get_backup_idle_timeout()

    assert result == 30 * 60
    assert "Invalid backup timeout 0, defaulting to 30" in caplog.text


# ---------------------------------------------------------------------------
# Tests for _get_env_bool / get_env_bool (Issue #3744)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "val",
    ["1", "true", "TRUE", "True", "yes", "YES", "Yes", "on", "ON", "  true  ", " 1 "],
)
def test_get_env_bool_truthy_values(monkeypatch, val):
    monkeypatch.setenv("TEST_FLAG", val)
    assert _get_env_bool("TEST_FLAG") is True
    assert get_env_bool("TEST_FLAG") is True


@pytest.mark.parametrize(
    "val",
    ["0", "false", "FALSE", "False", "no", "NO", "off", "OFF", "random_string", "none"],
)
def test_get_env_bool_falsy_values(monkeypatch, val):
    monkeypatch.setenv("TEST_FLAG", val)
    assert _get_env_bool("TEST_FLAG", default=True) is False
    assert get_env_bool("TEST_FLAG", default=True) is False


def test_get_env_bool_unset_uses_default(monkeypatch):
    monkeypatch.delenv("TEST_FLAG", raising=False)
    assert _get_env_bool("TEST_FLAG", default=False) is False
    assert _get_env_bool("TEST_FLAG", default=True) is True


def test_get_env_bool_empty_or_whitespace_uses_default(monkeypatch):
    monkeypatch.setenv("TEST_FLAG", "")
    assert _get_env_bool("TEST_FLAG", default=False) is False
    assert _get_env_bool("TEST_FLAG", default=True) is True

    monkeypatch.setenv("TEST_FLAG", "   ")
    assert _get_env_bool("TEST_FLAG", default=False) is False
    assert _get_env_bool("TEST_FLAG", default=True) is True


# ---------------------------------------------------------------------------
# Tests for _get_env_int / get_env_int bounds enforcement (Issue #3750)
# ---------------------------------------------------------------------------


def test_get_env_int_valid_value(monkeypatch):
    monkeypatch.setenv("TEST_INT_VAR", "42")
    assert _get_env_int("TEST_INT_VAR", default=10) == 42
    assert get_env_int("TEST_INT_VAR", default=10) == 42


def test_get_env_int_valid_with_bounds(monkeypatch):
    monkeypatch.setenv("TEST_INT_VAR", "50")
    assert _get_env_int("TEST_INT_VAR", default=10, min_val=0, max_val=100) == 50
    assert get_env_int("TEST_INT_VAR", default=10, min_val=0, max_val=100) == 50


def test_get_env_int_missing_variable_returns_default(monkeypatch):
    monkeypatch.delenv("TEST_INT_VAR", raising=False)
    assert _get_env_int("TEST_INT_VAR", default=15) == 15
    assert get_env_int("TEST_INT_VAR", default=15) == 15


@pytest.mark.parametrize("invalid_val", ["", "   ", "abc", "12.34", "NaN", "null", "none"])
def test_get_env_int_non_numeric_returns_default(monkeypatch, invalid_val):
    monkeypatch.setenv("TEST_INT_VAR", invalid_val)
    assert _get_env_int("TEST_INT_VAR", default=25) == 25
    assert get_env_int("TEST_INT_VAR", default=25) == 25


def test_get_env_int_less_than_min_val_returns_default(monkeypatch):
    monkeypatch.setenv("TEST_INT_VAR", "-5")
    assert _get_env_int("TEST_INT_VAR", default=10, min_val=0) == 10
    assert get_env_int("TEST_INT_VAR", default=10, min_val=0) == 10

    monkeypatch.setenv("TEST_INT_VAR", "4")
    assert _get_env_int("TEST_INT_VAR", default=20, min_val=5) == 20
    assert get_env_int("TEST_INT_VAR", default=20, min_val=5) == 20


def test_get_env_int_greater_than_max_val_returns_default(monkeypatch):
    monkeypatch.setenv("TEST_INT_VAR", "105")
    assert _get_env_int("TEST_INT_VAR", default=50, max_val=100) == 50
    assert get_env_int("TEST_INT_VAR", default=50, max_val=100) == 50

    monkeypatch.setenv("TEST_INT_VAR", "500")
    assert _get_env_int("TEST_INT_VAR", default=30, min_val=1, max_val=100) == 30
    assert get_env_int("TEST_INT_VAR", default=30, min_val=1, max_val=100) == 30


def test_get_env_int_exact_boundaries(monkeypatch):
    monkeypatch.setenv("TEST_INT_VAR", "10")
    assert _get_env_int("TEST_INT_VAR", default=0, min_val=10, max_val=20) == 10

    monkeypatch.setenv("TEST_INT_VAR", "20")
    assert _get_env_int("TEST_INT_VAR", default=0, min_val=10, max_val=20) == 20


# ---------------------------------------------------------------------------
# Tests for _get_env_bool (Issue #3749)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("truthy_val", ["true", "True", "TRUE", "1", "yes", "Yes", "YES"])
def test_get_env_bool_truthy_values(monkeypatch, truthy_val):
    monkeypatch.setenv("TEST_BOOL_VAR", truthy_val)
    assert _get_env_bool("TEST_BOOL_VAR", default=False) is True


@pytest.mark.parametrize("falsy_val", ["false", "False", "FALSE", "0", "no", "No", "NO"])
def test_get_env_bool_falsy_values(monkeypatch, falsy_val):
    monkeypatch.setenv("TEST_BOOL_VAR", falsy_val)
    assert _get_env_bool("TEST_BOOL_VAR", default=True) is False


def test_get_env_bool_missing_variable_returns_default(monkeypatch):
    monkeypatch.delenv("TEST_BOOL_VAR", raising=False)
    assert _get_env_bool("TEST_BOOL_VAR", default=True) is True
    assert _get_env_bool("TEST_BOOL_VAR", default=False) is False


def test_get_env_bool_empty_string_returns_default(monkeypatch):
    monkeypatch.setenv("TEST_BOOL_VAR", "")
    assert _get_env_bool("TEST_BOOL_VAR", default=True) is True
    assert _get_env_bool("TEST_BOOL_VAR", default=False) is False


def test_get_env_bool_whitespace_only_returns_default(monkeypatch):
    monkeypatch.setenv("TEST_BOOL_VAR", "   ")
    assert _get_env_bool("TEST_BOOL_VAR", default=True) is True


def test_get_env_bool_unrecognized_value_returns_default(monkeypatch):
    monkeypatch.setenv("TEST_BOOL_VAR", "maybe")
    assert _get_env_bool("TEST_BOOL_VAR", default=True) is True
    assert _get_env_bool("TEST_BOOL_VAR", default=False) is False
