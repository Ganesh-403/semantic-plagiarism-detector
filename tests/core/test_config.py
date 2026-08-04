import os
import json
import tempfile
import importlib.util

# Load config module directly without triggering package __init__ files
config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'core', 'config.py')
spec = importlib.util.spec_from_file_location("config_module", config_path)
config_module = importlib.util.module_from_spec(spec)
import sys
sys.modules["config_module"] = config_module
spec.loader.exec_module(config_module)

BrandingConfig = config_module.BrandingConfig
load_branding_config = config_module.load_branding_config
validate_hex_color = config_module.validate_hex_color
DEFAULT_BRAND_COLOR = config_module.DEFAULT_BRAND_COLOR
DEFAULT_LOGO_PATH = config_module.DEFAULT_LOGO_PATH


def test_validate_hex_color():
    """Test hex color validation."""
    # Valid colors
    assert validate_hex_color("#1e3a8a") is True
    assert validate_hex_color("#ABC") is True
    assert validate_hex_color("#123456") is True
    assert validate_hex_color("#abcdef") is True

    # Invalid colors
    assert validate_hex_color("1e3a8a") is False
    assert validate_hex_color("#1234567") is False
    assert validate_hex_color("#12") is False
    assert validate_hex_color("#gggggg") is False
    assert validate_hex_color(None) is False
    assert validate_hex_color("") is False


def test_branding_config_from_dict_valid():
    """Test BrandingConfig creation from valid dictionary."""
    valid_data = {"brand_color": "#ff0000", "logo_path": "/path/to/logo.png"}
    config = BrandingConfig.from_dict(valid_data)
    assert config.brand_color == "#ff0000"
    assert config.logo_path == "/path/to/logo.png"


def test_branding_config_from_dict_invalid_color():
    """Test BrandingConfig with invalid color uses default."""
    invalid_color_data = {"brand_color": "invalid", "logo_path": None}
    config = BrandingConfig.from_dict(invalid_color_data)
    assert config.brand_color == DEFAULT_BRAND_COLOR


def test_branding_config_from_dict_missing_fields():
    """Test BrandingConfig with missing fields uses defaults."""
    partial_data = {}
    config = BrandingConfig.from_dict(partial_data)
    assert config.brand_color == DEFAULT_BRAND_COLOR
    assert config.logo_path == DEFAULT_LOGO_PATH


def test_branding_config_from_dict_invalid_logo_type():
    """Test BrandingConfig with invalid logo type uses default."""
    invalid_logo_data = {"brand_color": "#ff0000", "logo_path": 123}
    config = BrandingConfig.from_dict(invalid_logo_data)
    assert config.logo_path == DEFAULT_LOGO_PATH


def test_load_valid_config():
    """Test loading a valid configuration file."""
    valid_config = {
        "brand_color": "#2ecc71",
        "logo_path": "/custom/logo.png"
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(valid_config, f)
        temp_path = f.name

    try:
        config = load_branding_config(temp_path)
        assert config.brand_color == "#2ecc71"
        assert config.logo_path == "/custom/logo.png"
    finally:
        os.unlink(temp_path)


def test_load_invalid_json():
    """Test loading invalid JSON falls back to defaults."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write("{ invalid json }")
        temp_path = f.name

    try:
        config = load_branding_config(temp_path)
        assert config.brand_color == DEFAULT_BRAND_COLOR
        assert config.logo_path == DEFAULT_LOGO_PATH
    finally:
        os.unlink(temp_path)


def test_load_invalid_schema():
    """Test loading invalid schema falls back to defaults."""
    invalid_schema_config = {
        "brand_color": "not-a-color",
        "logo_path": 12345
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(invalid_schema_config, f)
        temp_path = f.name

    try:
        config = load_branding_config(temp_path)
        assert config.brand_color == DEFAULT_BRAND_COLOR
        assert config.logo_path == DEFAULT_LOGO_PATH
    finally:
        os.unlink(temp_path)


def test_load_missing_file():
    """Test loading missing file falls back to defaults."""
    config = load_branding_config("/nonexistent/path/branding_config.json")
    assert config.brand_color == DEFAULT_BRAND_COLOR
    assert config.logo_path == DEFAULT_LOGO_PATH


def test_load_empty_json():
    """Test loading empty JSON falls back to defaults."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({}, f)
        temp_path = f.name

    try:
        config = load_branding_config(temp_path)
        assert config.brand_color == DEFAULT_BRAND_COLOR
        assert config.logo_path == DEFAULT_LOGO_PATH
    finally:
        os.unlink(temp_path)


def test_actual_branding_config():
    """Test loading the actual branding_config.json from the project."""
    config_path = os.path.join(os.path.dirname(__file__), "..", "..", "branding_config.json")

    if os.path.exists(config_path):
        config = load_branding_config(config_path)
        # Verify it loads without error and has valid structure
        assert isinstance(config, BrandingConfig)
        assert validate_hex_color(config.brand_color)


def test_to_dict():
    """Test BrandingConfig to_dict conversion."""
    config = BrandingConfig(brand_color="#ff0000", logo_path="/path/to/logo.png")
    data = config.to_dict()

    assert data["brand_color"] == "#ff0000"
    assert data["logo_path"] == "/path/to/logo.png"


def test_get_allowed_webhook_domains(monkeypatch):
    """Test get_allowed_webhook_domains under various environment configurations."""
    from src.core.app_config import get_allowed_webhook_domains

    # Unset env -> returns empty list
    monkeypatch.delenv("ALLOWED_WEBHOOK_DOMAINS", raising=False)
    assert get_allowed_webhook_domains() == []

    # Whitespace or empty -> returns empty list
    monkeypatch.setenv("ALLOWED_WEBHOOK_DOMAINS", "   ")
    assert get_allowed_webhook_domains() == []

    # Single domain
    monkeypatch.setenv("ALLOWED_WEBHOOK_DOMAINS", "hooks.slack.com")
    assert get_allowed_webhook_domains() == ["hooks.slack.com"]

    # Multiple domains with spaces and mixed case
    monkeypatch.setenv("ALLOWED_WEBHOOK_DOMAINS", " hooks.slack.com, Discord.com , EXAMPLE.org ")
    assert get_allowed_webhook_domains() == ["hooks.slack.com", "discord.com", "example.org"]
