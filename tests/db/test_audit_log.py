"""
Tests for src.db.audit_log
----------------------------
Covers schema initialisation, event recording, querying with filters,
FTS5 full-text search, action/actor counts, CSV/JSON export, and purge.
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from src.db.audit_log import (
    ACTION_DOCUMENT_DELETE,
    ACTION_DOCUMENT_UPLOAD,
    ACTION_SCAN_RUN,
    ACTION_USER_CREATE,
    ACTION_USER_LOGIN,
    AuditEntry,
    AuditFilter,
    close_connections,
    configure_db_path,
    count_entries,
    export_entries_csv,
    export_entries_json,
    get_action_counts,
    get_actor_counts,
    get_entry,
    get_recent_activity,
    init_audit_db,
    query_entries,
    record_document_delete,
    record_document_upload,
    record_event,
    record_incident_review,
    record_scan_run,
    record_threshold_change,
    record_user_action,
    record_export,
    record_document_restore,
    purge_old_entries,
)


@pytest.fixture(autouse=True)
def _setup_audit_db(tmp_path):
    """Create a fresh temp DB for each test and clean up after."""
    db_path = str(tmp_path / "test_audit.db")
    configure_db_path(db_path)
    init_audit_db()
    yield
    close_connections()


# ── Schema ────────────────────────────────────────────────────────────────────


class TestSchemaInit:
    def test_creates_tables(self):
        import sqlite3
        from src.db.audit_log import _DB_PATH
        conn = sqlite3.connect(_DB_PATH)
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        conn.close()
        assert "audit_log" in tables

    def test_idempotent(self):
        # Calling init twice should not raise
        init_audit_db()
        init_audit_db()


# ── Recording events ─────────────────────────────────────────────────────────


class TestRecordEvent:
    def test_returns_uuid(self):
        eid = record_event("test.action", "admin", "target", "detail")
        assert isinstance(eid, str)
        assert len(eid) == 36  # UUID format

    def test_stores_metadata(self):
        eid = record_event("test.action", "admin", metadata={"key": "value"})
        entry = get_entry(eid)
        assert entry is not None
        meta = json.loads(entry.metadata_json)
        assert meta["key"] == "value"

    def test_record_document_upload(self):
        eid = record_document_upload("essay.pdf", "teacher1")
        entry = get_entry(eid)
        assert entry is not None
        assert entry.action == ACTION_DOCUMENT_UPLOAD
        assert entry.target == "essay.pdf"
        assert entry.actor == "teacher1"

    def test_record_document_delete_soft(self):
        eid = record_document_delete("old.pdf", "admin", soft=True)
        entry = get_entry(eid)
        assert entry is not None
        assert entry.action == ACTION_DOCUMENT_DELETE
        meta = json.loads(entry.metadata_json)
        assert meta["soft_delete"] is True

    def test_record_document_delete_hard(self):
        eid = record_document_delete("old.pdf", "admin", soft=False)
        entry = get_entry(eid)
        meta = json.loads(entry.metadata_json)
        assert meta["soft_delete"] is False

    def test_record_document_restore(self):
        eid = record_document_restore("essay.pdf", "admin")
        entry = get_entry(eid)
        assert entry is not None
        assert entry.target == "essay.pdf"

    def test_record_scan_run(self):
        eid = record_scan_run("admin", document_count=10, flagged_count=2, threshold=0.59)
        entry = get_entry(eid)
        assert entry is not None
        assert entry.action == ACTION_SCAN_RUN
        meta = json.loads(entry.metadata_json)
        assert meta["document_count"] == 10
        assert meta["flagged_count"] == 2

    def test_record_incident_review(self):
        eid = record_incident_review("inc-123", "teacher1", "Resolved", "a.pdf", "b.pdf")
        entry = get_entry(eid)
        assert entry is not None
        assert entry.target == "inc-123"
        meta = json.loads(entry.metadata_json)
        assert meta["new_status"] == "Resolved"

    def test_record_threshold_change(self):
        eid = record_threshold_change("admin", 0.59, 0.65)
        entry = get_entry(eid)
        meta = json.loads(entry.metadata_json)
        assert meta["old_threshold"] == 0.59
        assert meta["new_threshold"] == 0.65

    def test_record_user_action(self):
        eid = record_user_action(ACTION_USER_CREATE, "admin", "newuser")
        entry = get_entry(eid)
        assert entry is not None
        assert entry.actor == "admin"
        assert entry.target == "newuser"

    def test_record_export(self):
        eid = record_export("admin", format="html", scope="full")
        entry = get_entry(eid)
        assert entry is not None
        meta = json.loads(entry.metadata_json)
        assert meta["format"] == "html"


# ── Querying ──────────────────────────────────────────────────────────────────


class TestQueryEntries:
    def _seed_entries(self):
        record_event("action.a", "alice", "doc1", "first event")
        record_event("action.b", "bob", "doc2", "second event")
        record_event("action.a", "alice", "doc3", "third event")
        record_event("action.c", "charlie", "doc1", "fourth event")

    def test_returns_all(self):
        self._seed_entries()
        entries = query_entries()
        assert len(entries) == 4

    def test_filter_by_action(self):
        self._seed_entries()
        entries = query_entries(AuditFilter(action="action.a"))
        assert len(entries) == 2
        assert all(e.action == "action.a" for e in entries)

    def test_filter_by_actor(self):
        self._seed_entries()
        entries = query_entries(AuditFilter(actor="bob"))
        assert len(entries) == 1
        assert entries[0].actor == "bob"

    def test_filter_by_target(self):
        self._seed_entries()
        entries = query_entries(AuditFilter(target="doc1"))
        assert len(entries) == 2

    def test_search_text(self):
        self._seed_entries()
        entries = query_entries(AuditFilter(search_text="third"))
        assert len(entries) == 1

    def test_limit_offset(self):
        self._seed_entries()
        entries = query_entries(AuditFilter(limit=2, offset=0))
        assert len(entries) == 2
        entries_page2 = query_entries(AuditFilter(limit=2, offset=2))
        assert len(entries_page2) == 2
        # Ensure no overlap
        ids1 = {e.entry_id for e in entries}
        ids2 = {e.entry_id for e in entries_page2}
        assert ids1.isdisjoint(ids2)

    def test_date_range(self):
        self._seed_entries()
        entries = query_entries(AuditFilter(start_date="2099-01-01"))
        assert len(entries) == 0  # No future dates

    def test_combined_filters(self):
        self._seed_entries()
        entries = query_entries(AuditFilter(action="action.a", actor="alice"))
        assert len(entries) == 2


# ── Counts ────────────────────────────────────────────────────────────────────


class TestCounts:
    def _seed(self):
        record_event("action.x", "u1")
        record_event("action.x", "u2")
        record_event("action.y", "u1")

    def test_count_entries(self):
        self._seed()
        assert count_entries() == 3

    def test_count_with_filter(self):
        self._seed()
        assert count_entries(AuditFilter(action="action.x")) == 2

    def test_get_action_counts(self):
        self._seed()
        counts = get_action_counts()
        assert counts["action.x"] == 2
        assert counts["action.y"] == 1

    def test_get_actor_counts(self):
        self._seed()
        actors = get_actor_counts()
        assert len(actors) == 2
        assert actors[0]["actor"] == "u1"
        assert actors[0]["count"] == 2


# ── Recent activity ───────────────────────────────────────────────────────────


class TestRecentActivity:
    def test_returns_entries(self):
        record_event("a", "u1")
        record_event("b", "u2")
        recent = get_recent_activity(limit=5)
        assert len(recent) == 2
        # Most recent first
        assert recent[0].action == "b"

    def test_empty_db(self):
        recent = get_recent_activity()
        assert recent == []


# ── Export ────────────────────────────────────────────────────────────────────


class TestExport:
    def _entries(self):
        record_event("action.x", "u1", "doc1", "detail here")
        record_event("action.y", "u2", "doc2", "other detail")
        return query_entries()

    def test_csv_export(self):
        entries = self._entries()
        csv_out = export_entries_csv(entries)
        assert "entry_id" in csv_out
        assert "doc1" in csv_out

    def test_json_export(self):
        entries = self._entries()
        json_out = export_entries_json(entries)
        data = json.loads(json_out)
        assert len(data) == 2
        assert data[0]["action"] == "action.y"  # Most recent first


# ── Purge ─────────────────────────────────────────────────────────────────────


class TestPurge:
    def test_purge_old(self):
        # Record entries then manipulate timestamps to be old
        record_event("test.action", "admin")
        import sqlite3
        from src.db.audit_log import _DB_PATH
        conn = sqlite3.connect(_DB_PATH)
        conn.execute("UPDATE audit_log SET timestamp = '2020-01-01T00:00:00'")
        conn.commit()
        conn.close()

        deleted = purge_old_entries(days=30)
        assert deleted == 1
        assert count_entries() == 0

    def test_purge_keeps_recent(self):
        record_event("test.action", "admin")
        deleted = purge_old_entries(days=365)
        assert deleted == 0
        assert count_entries() == 1


# ── get_entry ─────────────────────────────────────────────────────────────────


class TestGetEntry:
    def test_existing(self):
        eid = record_event("x", "u")
        entry = get_entry(eid)
        assert entry is not None
        assert entry.entry_id == eid

    def test_nonexistent(self):
        assert get_entry("nonexistent-id") is None


# ── AuditEntry dataclass ─────────────────────────────────────────────────────


class TestAuditEntry:
    def test_to_dict(self):
        entry = AuditEntry(
            entry_id="id-1", timestamp="2025-08-01T10:00:00",
            action="test", actor="admin", target="doc",
            detail="detail", metadata_json='{"k": "v"}',
        )
        d = entry.to_dict()
        assert d["entry_id"] == "id-1"
        assert d["metadata"]["k"] == "v"

    def test_to_dict_empty_metadata(self):
        entry = AuditEntry(
            entry_id="id-2", timestamp="2025-08-01T10:00:00",
            action="test", actor="admin", target="",
            detail="", metadata_json="{}",
        )
        d = entry.to_dict()
        assert d["metadata"] == {}
