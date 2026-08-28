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

from pathlib import Path

SOURCE = Path("src/security/ssrf_protector.py")
TESTS = Path("tests/security/test_ssrf_protector.py")


def test_required_cidr_helper_exists():
    source = SOURCE.read_text(encoding="utf-8")

    assert "def is_ip_in_cidr_block(" in source
    assert "ip_str: str" in source
    assert "cidr_block: str" in source
    assert ") -> bool:" in source
    assert "ipaddress.ip_network(" in source


def test_all_required_cidr_blocks_are_configured():
    source = SOURCE.read_text(encoding="utf-8")

    for cidr in (
        "127.0.0.0/8",
        "10.0.0.0/8",
        "172.16.0.0/12",
        "192.168.0.0/16",
    ):
        assert cidr in source


def test_url_validation_uses_cidr_helper():
    source = SOURCE.read_text(encoding="utf-8")

    assert "for cidr_block in " "cls.RESTRICTED_IPV4_CIDR_BLOCKS" in source
    assert "is_ip_in_cidr_block(ip_str, cidr_block)" in source
    tests = TESTS.read_text(encoding="utf-8")
    assert "test_validate_url_safety_integrates_required_cidr_filter" in tests
