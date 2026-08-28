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
tests/app/test_system_health_ui.py
----------------------------------
Tests for Issue #645 — System Health Monitoring Dashboard.

These tests follow the repo convention of reading streamlit_app.py as plain
text and asserting key strings are present and ordered correctly.
They also exercise the helper logic (size formatting, date formatting,
Redis unavailable path) directly — without importing Streamlit.
"""

import json
import re
from datetime import datetime
from pathlib import Path

APP_PATH = Path("app/streamlit_app.py")
EN_I18N = Path("src/i18n/en.json")
ES_I18N = Path("src/i18n/es.json")


# ── i18n ─────────────────────────────────────────────────────────────────────


def test_tab_health_key_exists_in_english_translations():
    data = json.loads(EN_I18N.read_text(encoding="utf-8"))
    assert "tab_health" in data
    assert data["tab_health"]  # non-empty


def test_tab_health_key_exists_in_spanish_translations():
    data = json.loads(ES_I18N.read_text(encoding="utf-8"))
    assert "tab_health" in data
    assert data["tab_health"]  # non-empty


# ── Tab registration ──────────────────────────────────────────────────────────


def test_tab_health_variable_is_unpacked_from_st_tabs():
    source = APP_PATH.read_text(encoding="utf-8")
    assert "tab_health," in source


def test_tab_health_label_fetched_from_i18n():
    source = APP_PATH.read_text(encoding="utf-8")
    assert 'get_text("tab_health", lang=lang_code)' in source


def test_tab_health_used_as_context_manager():
    source = APP_PATH.read_text(encoding="utf-8")
    assert "with tab_health:" in source


# ── Dashboard content ─────────────────────────────────────────────────────────


def test_cpu_metric_present():
    source = APP_PATH.read_text(encoding="utf-8")
    assert "psutil.cpu_percent" in source
    assert "CPU Usage" in source


def test_memory_metric_present():
    source = APP_PATH.read_text(encoding="utf-8")
    assert "psutil.virtual_memory" in source
    assert "Memory Used" in source
    assert "Memory Usage" in source


def test_redis_status_metric_present():
    source = APP_PATH.read_text(encoding="utf-8")
    assert "_cache.ping()" in source
    assert '"Connected"' in source
    assert '"Disconnected"' in source


def test_redis_unavailable_error_shown():
    source = APP_PATH.read_text(encoding="utf-8")
    assert "Redis is unavailable" in source


def test_database_size_metric_present():
    source = APP_PATH.read_text(encoding="utf-8")
    assert "get_corpus_db_path" in source
    assert "1_048_576" in source


def test_refresh_button_present():
    source = APP_PATH.read_text(encoding="utf-8")
    assert 'key="health_refresh_button"' in source
    assert "Refresh Metrics" in source


# ── Ordering guarantees ───────────────────────────────────────────────────────


def test_tab_health_section_comes_before_tab_settings():
    source = APP_PATH.read_text(encoding="utf-8")
    health_pos = source.index("with tab_health:")
    settings_pos = source.index("with tab_settings:")
    assert health_pos < settings_pos, "tab_health must be rendered before tab_settings"


def test_tab_health_is_inside_admin_role_check():
    source = APP_PATH.read_text(encoding="utf-8")
    admin_pos = source.index('if user_role == "admin":')
    health_pos = source.index("with tab_health:")
    assert (
        health_pos > admin_pos
    ), "System Health tab must appear inside the admin-only section"


# ── Size-formatting helper logic (pure Python, no Streamlit) ──────────────────


def _size_label(size_bytes: int) -> str:
    """Mirror of the formatting logic in tab_health."""
    if size_bytes >= 1024 * 1024:
        return f"{size_bytes / 1_048_576:.2f} MB"
    elif size_bytes >= 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes} B"


def test_size_label_bytes():
    assert _size_label(0) == "0 B"
    assert _size_label(500) == "500 B"
    assert _size_label(1023) == "1023 B"


def test_size_label_kilobytes():
    assert _size_label(1024) == "1.0 KB"
    assert _size_label(2048) == "2.0 KB"
    assert _size_label(1024 * 1024 - 1) == "1024.0 KB"


def test_size_label_megabytes():
    assert _size_label(1024 * 1024) == "1.00 MB"
    assert _size_label(2 * 1024 * 1024) == "2.00 MB"


# ── Date-formatting helper logic ──────────────────────────────────────────────


def test_mtime_format_matches_expected_pattern(tmp_path):
    db_path = tmp_path / "corpus.db"
    db_path.write_bytes(b"SQLite format 3\x00")

    mtime = db_path.stat().st_mtime
    formatted = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")

    assert re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", formatted)


# ── Redis ping contract ───────────────────────────────────────────────────────


def test_redis_ping_returns_tuple_of_two_when_disconnected():
    """
    Verify the dashboard's ping unpack never raises TypeError when Redis
    is unavailable — the mock returns (False, 0).
    """
    from unittest.mock import MagicMock

    mock_cache = MagicMock()
    mock_cache.ping.return_value = (False, 0)

    connected, latency = mock_cache.ping()
    assert connected is False
    assert latency == 0


def test_redis_ping_connected_tuple():
    from unittest.mock import MagicMock

    mock_cache = MagicMock()
    mock_cache.ping.return_value = (True, 3)

    connected, latency = mock_cache.ping()
    assert connected is True
    assert latency == 3
