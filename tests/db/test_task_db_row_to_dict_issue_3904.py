"""Regression coverage for issue #3904.

``src/db/task_db.py`` defined ``_row_to_dict`` twice.

The first definition (line 191) is the one the module is built around.  It
wraps the row in ``JobRecord``, wraps the JSON columns in ``JsonPayload``, and
adds three backwards-compatibility aliases::

    d["attempts"]      = d.get("retry_count", 0)
    d["max_attempts"]  = d.get("max_retries", 3)
    d["error_message"] = d.get("error")

A second definition 280 lines further down, under a ``# ── Helpers ──`` banner,
silently replaced it with one that returned a plain ``dict`` and none of those
aliases.  Python binds the last definition, so every call site --
``get_job``, ``list_jobs``, ``claim_next_job`` -- got the stripped-down version:

* ``attempts``, ``max_attempts`` and ``error_message`` vanished from every job;
* ``JobRecord`` and ``JsonPayload`` became dead code, so ``job.id`` and
  ``JsonPayload``'s dict-like access were unreachable;
* the annotations lied -- ``create_job`` and ``get_job`` are declared
  ``-> JobRecord`` / ``-> Optional[JobRecord]`` but returned a bare ``dict``.

``src/core/batch_history.py:190`` already reads ``job_data.get("error_message")``
and was silently getting ``None`` for every failed job.

``TestOnlyOneDefinition`` is the guard that would have caught this at review
time; the remaining classes pin the behaviour that was lost.
"""

from __future__ import annotations

import ast
import json
import sqlite3
from pathlib import Path

import pytest

from src.db import task_db
from src.db.task_db import JobRecord, JsonPayload, _row_to_dict

TASK_DB_PATH = Path(__file__).resolve().parents[2] / "src" / "db" / "task_db.py"


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    """An isolated task-queue database file."""
    return tmp_path / "task_queue.db"


@pytest.fixture()
def row_factory():
    """Build a ``sqlite3.Row`` shaped like a ``task_jobs`` row."""

    def _make(**overrides) -> sqlite3.Row:
        values = {
            "id": "job-1",
            "status": "FAILED",
            "payload": '{"document": "a.pdf"}',
            "result": None,
            "error": "boom",
            "retry_count": 2,
            "max_retries": 5,
        }
        values.update(overrides)
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        columns = ", ".join(values)
        placeholders = ", ".join("?" for _ in values)
        conn.execute(f"CREATE TABLE t ({columns})")
        conn.execute(f"INSERT INTO t VALUES ({placeholders})", tuple(values.values()))
        return conn.execute("SELECT * FROM t").fetchone()

    return _make


class TestOnlyOneDefinition:
    """The structural guard -- a duplicate must never win silently again."""

    def test_row_to_dict_is_defined_exactly_once(self) -> None:
        tree = ast.parse(TASK_DB_PATH.read_text(encoding="utf-8-sig"))
        definitions = [
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "_row_to_dict"
        ]
        assert len(definitions) == 1, (
            "_row_to_dict is defined "
            f"{len(definitions)} times (lines {[d.lineno for d in definitions]}); "
            "the last definition silently shadows the others"
        )

    def test_no_module_scope_name_is_defined_twice(self) -> None:
        """Generalise the check to every top-level function and class."""
        tree = ast.parse(TASK_DB_PATH.read_text(encoding="utf-8-sig"))
        seen: dict[str, int] = {}
        duplicates: list[str] = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name in seen:
                    duplicates.append(
                        f"{node.name!r} at line {node.lineno} shadows line {seen[node.name]}"
                    )
                seen[node.name] = node.lineno
        assert not duplicates, "duplicate top-level definitions in task_db.py:\n" + "\n".join(
            duplicates
        )

    def test_the_surviving_definition_is_the_jobrecord_one(self) -> None:
        """Guard against "fixing" the duplicate by deleting the wrong one."""
        source = TASK_DB_PATH.read_text(encoding="utf-8-sig")
        tree = ast.parse(source)
        definition = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_row_to_dict"
        )
        body_source = ast.get_source_segment(source, definition) or ""
        assert "JobRecord(" in body_source, "the surviving _row_to_dict must build a JobRecord"
        assert "JsonPayload(" in body_source, "the surviving _row_to_dict must wrap JSON columns"
        for alias in ("attempts", "max_attempts", "error_message"):
            assert f'"{alias}"' in body_source, f"the surviving _row_to_dict must set {alias!r}"


class TestRowToDictReturnShape:
    """The behaviour the shadowing definition removed."""

    def test_returns_a_jobrecord(self, row_factory) -> None:
        result = _row_to_dict(row_factory())
        assert isinstance(result, JobRecord)
        assert isinstance(result, dict)  # JobRecord subclasses dict

    def test_exposes_the_compatibility_aliases(self, row_factory) -> None:
        result = _row_to_dict(row_factory(retry_count=2, max_retries=5, error="boom"))
        assert result["attempts"] == 2
        assert result["max_attempts"] == 5
        assert result["error_message"] == "boom"

    def test_aliases_track_the_underlying_columns(self, row_factory) -> None:
        result = _row_to_dict(row_factory(retry_count=7, max_retries=9, error=None))
        assert result["attempts"] == 7
        assert result["max_attempts"] == 9
        assert result["error_message"] is None

    def test_original_columns_are_preserved(self, row_factory) -> None:
        """The aliases are additive -- they must not replace the real columns."""
        result = _row_to_dict(row_factory(retry_count=2, max_retries=5))
        assert result["retry_count"] == 2
        assert result["max_retries"] == 5

    def test_jobrecord_id_property(self, row_factory) -> None:
        """``JobRecord.id`` was unreachable while the plain dict won."""
        result = _row_to_dict(row_factory(id="job-42"))
        assert result.id == "job-42"
        assert str(result) == "job-42"

    def test_payload_is_a_jsonpayload(self, row_factory) -> None:
        result = _row_to_dict(row_factory(payload='{"document": "a.pdf"}'))
        payload = result["payload"]
        assert isinstance(payload, JsonPayload)
        # It is still a str, so code that logs or stores it keeps working ...
        assert isinstance(payload, str)
        assert json.loads(payload) == {"document": "a.pdf"}
        # ... and it is also indexable like the decoded object.
        assert payload["document"] == "a.pdf"
        assert payload.get("document") == "a.pdf"
        assert payload.get("missing", "fallback") == "fallback"
        assert "document" in payload

    def test_null_result_column_is_left_alone(self, row_factory) -> None:
        result = _row_to_dict(row_factory(result=None))
        assert result["result"] is None

    def test_result_column_is_wrapped_when_present(self, row_factory) -> None:
        result = _row_to_dict(row_factory(result='{"score": 0.91}'))
        assert isinstance(result["result"], JsonPayload)
        assert result["result"]["score"] == 0.91

    def test_malformed_json_does_not_raise(self, row_factory) -> None:
        """``JsonPayload`` degrades to an empty mapping rather than blowing up."""
        result = _row_to_dict(row_factory(payload="not json at all"))
        payload = result["payload"]
        assert str(payload) == "not json at all"
        assert payload.get("anything") is None
        assert list(payload.keys()) == []


class TestJsonPayloadDirectly:
    """``JsonPayload`` was dead code under the shadowing definition."""

    def test_wraps_a_dict(self) -> None:
        payload = JsonPayload({"a": 1})
        assert json.loads(str(payload)) == {"a": 1}
        assert payload["a"] == 1
        assert dict(payload.items()) == {"a": 1}
        assert list(payload.values()) == [1]

    def test_wraps_a_list(self) -> None:
        payload = JsonPayload([1, 2, 3])
        assert payload[0] == 1
        assert 2 in payload

    def test_wraps_a_json_string(self) -> None:
        payload = JsonPayload('{"a": 1}')
        assert payload["a"] == 1

    def test_empty_value_is_an_empty_mapping(self) -> None:
        payload = JsonPayload("")
        assert str(payload) == ""
        assert payload.get("a") is None


class TestEndToEndThroughThePublicApi:
    """The aliases must survive the real ``create_job`` / ``get_job`` path."""

    def test_create_job_returns_a_jobrecord_with_aliases(self, db_path) -> None:
        job = task_db.create_job({"document": "a.pdf"}, max_retries=4, db_path=db_path)
        assert isinstance(job, JobRecord)
        assert job["attempts"] == 0
        assert job["max_attempts"] == 4
        assert job["error_message"] is None
        assert job["payload"]["document"] == "a.pdf"

    def test_get_job_round_trips(self, db_path) -> None:
        created = task_db.create_job({"document": "b.pdf"}, db_path=db_path)
        fetched = task_db.get_job(created.id, db_path=db_path)
        assert fetched is not None
        assert isinstance(fetched, JobRecord)
        assert fetched.id == created.id
        assert fetched["attempts"] == 0
        assert fetched["max_attempts"] == 3

    def test_list_jobs_returns_jobrecords(self, db_path) -> None:
        task_db.create_job({"document": "a.pdf"}, db_path=db_path)
        task_db.create_job({"document": "b.pdf"}, db_path=db_path)
        jobs = task_db.list_jobs(db_path=db_path)
        assert len(jobs) == 2
        for job in jobs:
            assert isinstance(job, JobRecord)
            assert "attempts" in job
            assert "max_attempts" in job
            assert "error_message" in job

    def test_claim_next_job_returns_a_jobrecord(self, db_path) -> None:
        created = task_db.create_job({"document": "a.pdf"}, db_path=db_path)
        claimed = task_db.claim_next_job("worker-1", db_path=db_path)
        assert claimed is not None
        assert isinstance(claimed, JobRecord)
        assert claimed.id == created.id
        assert claimed["attempts"] == 0

    def test_error_message_is_populated_after_a_failure(self, db_path) -> None:
        """The exact consumer in src/core/batch_history.py:190.

        Under the shadowing definition this returned ``None`` for every job,
        so a failed batch reported no reason.
        """
        created = task_db.create_job({"document": "a.pdf"}, db_path=db_path)
        task_db.mark_failed(created.id, "parser exploded", db_path=db_path)

        job = task_db.get_job(created.id, db_path=db_path)
        assert job is not None
        assert job["error"] == "parser exploded"
        assert job.get("error_message") == "parser exploded"


class TestTypingImport:
    """``Union`` is used 14 times in this module's annotations."""

    def test_union_is_imported(self) -> None:
        assert hasattr(task_db, "Union"), (
            "task_db uses Union in its annotations but never imports it; "
            "'from __future__ import annotations' hides this until something "
            "evaluates the annotations"
        )

    def test_annotations_actually_resolve(self) -> None:
        """``get_type_hints`` is what turns a missing import into a NameError."""
        from typing import get_type_hints

        for func in (task_db.create_job, task_db.get_job, task_db._row_to_dict):
            hints = get_type_hints(func)
            assert hints, f"no resolvable type hints on {func.__name__}"
