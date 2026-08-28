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
