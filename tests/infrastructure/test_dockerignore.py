"""
test_dockerignore.py
--------------------
Tests for .dockerignore to ensure sensitive and unwanted files are excluded
from Docker builds (Issue #2943).
"""

from pathlib import Path


def test_dockerignore_exists_and_contains_required_patterns():
    """Ensure .dockerignore exists and explicitly excludes required patterns."""
    repo_root = Path(__file__).resolve().parents[2]
    dockerignore_path = repo_root / ".dockerignore"

    assert dockerignore_path.exists(), ".dockerignore file must exist in repo root"

    content = dockerignore_path.read_text(encoding="utf-8")
    lines = [
        line.strip()
        for line in content.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    # Required patterns according to acceptance criteria for Issue #2943
    required_patterns = [".env", ".git/", ".venv/", "*.sqlite"]

    for pattern in required_patterns:
        assert pattern in lines, (
            f"Required pattern '{pattern}' must be explicitly present in .dockerignore"
        )


def test_dockerignore_excludes_bytecode_and_secrets():
    """Ensure .dockerignore excludes pycache, env variants, and SQLite databases."""
    repo_root = Path(__file__).resolve().parents[2]
    dockerignore_path = repo_root / ".dockerignore"

    content = dockerignore_path.read_text(encoding="utf-8")
    lines = [
        line.strip()
        for line in content.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    expected_exclusions = [
        ".env",
        ".git/",
        ".venv/",
        "*.sqlite",
        "*.db",
        "__pycache__/",
    ]

    for exclusion in expected_exclusions:
        assert exclusion in lines, (
            f"Expected pattern '{exclusion}' should be in .dockerignore"
        )
