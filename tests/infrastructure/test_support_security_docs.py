"""
test_support_security_docs.py
-----------------------------
Tests verifying open-source governance and security documentation compliance.
Ensures SUPPORT.md contains a dedicated Security Vulnerabilities policy directing users
to email security@domain.com instead of filing public GitHub issues.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def test_support_md_has_security_vulnerabilities_section():
    """Verify SUPPORT.md has a Security Vulnerabilities section with private email alias."""
    support_path = REPO_ROOT / "SUPPORT.md"
    assert support_path.exists(), "SUPPORT.md file must exist in repo root"

    content = support_path.read_text(encoding="utf-8")

    assert "Security Vulnerabilities" in content
    assert "security@domain.com" in content
    assert "SECURITY.md" in content
    assert (
        "please do NOT open a public GitHub issue" in content
        or "do NOT open a public" in content
    )


def test_security_md_has_security_email_alias():
    """Verify SECURITY.md contains the private security email alias."""
    security_path = REPO_ROOT / "SECURITY.md"
    assert security_path.exists(), "SECURITY.md file must exist in repo root"

    content = security_path.read_text(encoding="utf-8")

    assert "security@domain.com" in content
