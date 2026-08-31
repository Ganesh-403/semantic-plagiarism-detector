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


def test_support_md_links_and_issue_templates_exist():
    """Verify all relative markdown links and issue templates referenced in SUPPORT.md exist."""
    import re
    from urllib.parse import urlparse

    support_path = REPO_ROOT / "SUPPORT.md"
    assert support_path.exists(), "SUPPORT.md must exist in repo root"

    content = support_path.read_text(encoding="utf-8")

    # Match all markdown links [text](url)
    link_pattern = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
    matches = link_pattern.findall(content)

    assert len(matches) > 0, "SUPPORT.md must contain markdown links"

    for link_text, link_url in matches:
        if link_url.startswith("mailto:"):
            assert "@" in link_url
            continue

        parsed = urlparse(link_url)
        if parsed.scheme in ("http", "https"):
            assert "github.com" in parsed.netloc
            assert "semantic-plagiarism-detector" in parsed.path
        else:
            # Relative file link
            target_path = (REPO_ROOT / link_url).resolve()
            assert target_path.exists(), f"Target link '{link_url}' referenced in SUPPORT.md does not exist at {target_path}"

