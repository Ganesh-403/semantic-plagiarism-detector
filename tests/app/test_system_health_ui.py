"""Behavioral tests for the administrator health panel."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest
from app.views import health_view as health


def render(role):
    from app.views.health_view import render_health_view
    render_health_view(role)


@pytest.fixture
def metrics(monkeypatch, tmp_path):
    database = tmp_path / "corpus.db"
    database.write_bytes(b"x" * 2048)
    monkeypatch.setattr(health, "get_corpus_db_path", lambda: str(database))
    monkeypatch.setattr(health.psutil, "cpu_percent", lambda **kw: 12.5)
    monkeypatch.setattr(health.psutil, "virtual_memory", lambda: SimpleNamespace(percent=25, used=1048576))
    monkeypatch.setattr(health, "get_cache", lambda: SimpleNamespace(ping=lambda: (True, 3)))
    return database


@pytest.mark.parametrize("language", ["en", "es"])
def test_health_navigation_translation(language):
    assert json.loads(Path(f"src/i18n/{language}.json").read_text(encoding="utf-8"))["tab_health"]


@pytest.mark.parametrize("size,label", [(0,"0 B"),(500,"500 B"),(1024,"1.0 KB"),(2048,"2.0 KB"),(1048576,"1.00 MB"),(2097152,"2.00 MB")])
def test_storage_size(size, label):
    assert health.format_storage_size(size) == label


def test_collects_cpu_memory_database_and_redis(metrics):
    assert health.collect_system_health() == {"cpu":12.5,"memory_percent":25,"memory_used":1048576,"redis_connected":True,"redis_latency":3,"database_bytes":2048}


def test_missing_database_is_zero(metrics):
    metrics.unlink()
    assert health.collect_system_health()["database_bytes"] == 0


def test_redis_failure_is_a_disconnected_metric(metrics, monkeypatch):
    def unavailable():
        raise ConnectionError("offline")
    monkeypatch.setattr(health, "get_cache", unavailable)
    assert health.collect_system_health()["redis_connected"] is False


def test_admin_dashboard_renders_metrics_and_refreshes(metrics):
    app = AppTest.from_function(render, args=("admin",)).run()
    assert not app.exception
    assert {metric.label: metric.value for metric in app.metric} == {"CPU Usage":"12.5%","Memory Usage":"25.0%","Database Size":"2.0 KB","Redis":"Connected"}
    app.button(key="health_refresh_button").click().run()
    assert not app.exception


def test_redis_outage_is_visible(metrics, monkeypatch):
    monkeypatch.setattr(health, "get_cache", lambda: SimpleNamespace(ping=lambda:(False,0)))
    app = AppTest.from_function(render, args=("admin",)).run()
    assert not app.exception
    assert "Redis is unavailable" in app.warning[0].value


@pytest.mark.parametrize("role", ["teacher", "analyst", ""])
def test_non_administrators_cannot_read_system_metrics(monkeypatch, role):
    def forbidden():
        pytest.fail("Non-administrator accessed system metrics")
    monkeypatch.setattr(health, "collect_system_health", forbidden)
    app = AppTest.from_function(render, args=(role,)).run()
    assert not app.exception and not app.metric
    assert "administrators" in app.info[0].value
