"""
test_pre_commit_secrets_detection_issue_2936.py
--------------------------------------------------
Unit tests for Issue #2936: Add secrets detection to pre-commit pipeline (.pre-commit-config.yaml).
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def test_pre_commit_config_contains_detect_secrets():
    """Verify .pre-commit-config.yaml includes the detect-secrets hook."""
    config_path = REPO_ROOT / ".pre-commit-config.yaml"
    assert config_path.exists(), ".pre-commit-config.yaml does not exist"

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    repos = config.get("repos", [])
    detect_secrets_repos = [
        r for r in repos if "detect-secrets" in r.get("repo", "")
    ]
    assert len(detect_secrets_repos) > 0, "No detect-secrets repository found in .pre-commit-config.yaml"

    hook_ids = [
        h.get("id") for r in detect_secrets_repos for h in r.get("hooks", [])
    ]
    assert "detect-secrets" in hook_ids, "detect-secrets hook id missing in .pre-commit-config.yaml"


def test_secrets_baseline_exists_and_valid_json():
    """Verify .secrets.baseline baseline file exists and is valid JSON."""
    baseline_path = REPO_ROOT / ".secrets.baseline"
    assert baseline_path.exists(), ".secrets.baseline file missing"

    with open(baseline_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "version" in data
    assert "plugins_used" in data
