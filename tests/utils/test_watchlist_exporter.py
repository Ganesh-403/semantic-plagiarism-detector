"""Tests for src.utils.watchlist_exporter."""

from __future__ import annotations

import csv
import json
import os
import tempfile

import pytest

from src.core.similarity_watchlist import (
    AlertTrigger,
    WatchlistAlert,
    WatchlistEntry,
    WatchlistStatus,
    WatchlistSummary,
    WatchlistType,
)
from src.utils.watchlist_exporter import (
    export_alerts_csv,
    export_entries_csv,
    export_watchlist_html,
    export_watchlist_json,
    export_watchlist_markdown,
)


def _make_entries():
    return [
        WatchlistEntry(
            entry_id=1,
            watchlist_type=WatchlistType.DOCUMENT,
            target="essay1.pdf",
            label="Monitor essay1",
            status=WatchlistStatus.ACTIVE,
            similarity_threshold=0.80,
            created_by="admin",
            created_at="2025-01-01T00:00:00",
        ),
        WatchlistEntry(
            entry_id=2,
            watchlist_type=WatchlistType.PAIR,
            target="essay1.pdf|essay2.pdf",
            label="Watch pair",
            status=WatchlistStatus.PAUSED,
            created_by="teacher1",
            created_at="2025-01-02T00:00:00",
        ),
    ]


def _make_alerts():
    return [
        WatchlistAlert(
            alert_id=1,
            entry_id=1,
            triggered_by=AlertTrigger.NEW_SCAN,
            matched_document="essay2.pdf",
            similarity_score=0.92,
            severity="High",
            scan_timestamp="2025-01-01T12:00:00",
            acknowledged=False,
            created_at="2025-01-01T12:00:00",
        ),
        WatchlistAlert(
            alert_id=2,
            entry_id=1,
            triggered_by=AlertTrigger.RESCAN,
            matched_document="essay3.pdf",
            similarity_score=0.75,
            severity="Medium",
            scan_timestamp="2025-01-02T12:00:00",
            acknowledged=True,
            created_at="2025-01-02T12:00:00",
        ),
    ]


def _make_summary():
    return WatchlistSummary(
        total_entries=2,
        active_entries=1,
        paused_entries=1,
        resolved_entries=0,
        total_alerts=2,
        unacknowledged_alerts=1,
        entries_by_type={"document": 1, "pair": 1},
        alerts_by_severity={"High": 1, "Medium": 1},
        recent_alerts=[],
    )


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------

class TestExportWatchlistJson:
    def test_creates_valid_json(self):
        entries = _make_entries()
        alerts = _make_alerts()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "watchlist.json")
            result = export_watchlist_json(entries, alerts, path)
            assert os.path.exists(result)
            with open(result) as f:
                data = json.load(f)
            assert data["entry_count"] == 2
            assert data["alert_count"] == 2
            assert len(data["entries"]) == 2
            assert len(data["alerts"]) == 2

    def test_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sub", "watchlist.json")
            export_watchlist_json([], [], path)
            assert os.path.exists(path)


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

class TestExportEntriesCsv:
    def test_creates_valid_csv(self):
        entries = _make_entries()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "entries.csv")
            result = export_entries_csv(entries, path)
            assert os.path.exists(result)
            with open(result) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert len(rows) == 2
            assert rows[0]["target"] == "essay1.pdf"

    def test_empty_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "entries.csv")
            export_entries_csv([], path)
            with open(path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert len(rows) == 0


class TestExportAlertsCsv:
    def test_creates_valid_csv(self):
        alerts = _make_alerts()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "alerts.csv")
            result = export_alerts_csv(alerts, path)
            assert os.path.exists(result)
            with open(result) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert len(rows) == 2
            assert rows[0]["severity"] == "High"


# ---------------------------------------------------------------------------
# Markdown export
# ---------------------------------------------------------------------------

class TestExportWatchlistMarkdown:
    def test_creates_markdown(self):
        entries = _make_entries()
        alerts = _make_alerts()
        summary = _make_summary()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "watchlist.md")
            result = export_watchlist_markdown(entries, alerts, summary, path)
            assert os.path.exists(result)
            with open(result) as f:
                content = f.read()
            assert "Similarity Watchlist Report" in content
            assert "essay1.pdf" in content
            assert "Summary" in content

    def test_custom_title(self):
        summary = _make_summary()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "watchlist.md")
            export_watchlist_markdown([], [], summary, path, title="My Report")
            with open(result_path if 'result_path' in dir() else path) as f:
                content = f.read()
            assert "My Report" in content


# ---------------------------------------------------------------------------
# HTML export
# ---------------------------------------------------------------------------

class TestExportWatchlistHtml:
    def test_creates_html(self):
        entries = _make_entries()
        alerts = _make_alerts()
        summary = _make_summary()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "watchlist.html")
            result = export_watchlist_html(entries, alerts, summary, path)
            assert os.path.exists(result)
            with open(result) as f:
                content = f.read()
            assert "<!DOCTYPE html>" in content
            assert "essay1.pdf" in content
            assert "0.92" in content

    def test_empty_data(self):
        summary = _make_summary()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "watchlist.html")
            export_watchlist_html([], [], summary, path)
            with open(path) as f:
                content = f.read()
            assert "No entries" in content
            assert "No alerts" in content

    def test_html_structure(self):
        entries = _make_entries()
        alerts = _make_alerts()
        summary = _make_summary()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "watchlist.html")
            export_watchlist_html(entries, alerts, summary, path, title="Dashboard")
            with open(path) as f:
                content = f.read()
            assert "Dashboard" in content
            assert "Watchlist Entries" in content
            assert "Alerts" in content
            assert "status-active" in content
            assert "sev-high" in content
