from pathlib import Path


INCIDENTS_PATH = Path("src/db/incidents.py")


def test_get_all_incidents_has_required_defaults():
    source = INCIDENTS_PATH.read_text(encoding="utf-8")

    assert "limit: int = 50" in source
    assert "offset: int = 0" in source


def test_query_uses_parameterized_limit_and_offset():
    source = INCIDENTS_PATH.read_text(encoding="utf-8")

    assert "LIMIT ? OFFSET ?" in source
    assert "(safe_limit, safe_offset)" in source


def test_total_count_helper_exists():
    source = INCIDENTS_PATH.read_text(encoding="utf-8")

    assert "def get_total_incidents_count(" in source
    assert "SELECT COUNT(*)" in source
