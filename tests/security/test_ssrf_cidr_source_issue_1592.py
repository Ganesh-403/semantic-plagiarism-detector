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

    assert "for cidr_block in cls.RESTRICTED_IPV4_CIDR_BLOCKS" in source
    assert "is_ip_in_cidr_block(ip_str, cidr_block)" in source
    tests = TESTS.read_text(encoding="utf-8")
    assert "test_validate_url_safety_integrates_required_cidr_filter" in tests
