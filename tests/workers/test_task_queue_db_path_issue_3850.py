"""
tests/workers/test_task_queue_db_path_issue_3850.py
----------------------------------------------------
Regression tests for Issue #3850.

``TaskQueue.__init__`` calls ``task_db.initialize_task_db(Path(db_path))`` but
``src/workers/task_queue.py`` never imported ``pathlib.Path``. Because the
module carries ``from __future__ import annotations`` the missing name stayed
invisible until that line ran, and it only runs on the ``db_path is not None``
branch — so ``TaskQueue()`` looked healthy while ``TaskQueue(db_path=...)``
raised::

    NameError: name 'Path' is not defined

``db_path`` is the only way to point the queue at a database other than
``DATA_DIR/task_queue.db``, which also made the class impossible to test
against a ``tmp_path``. These tests drive the full job lifecycle through an
explicitly-pathed queue — the scenario the bug blocked.
"""

from __future__ import annotations

import ast
import pathlib
import queue as queue_module
import sqlite3
from pathlib import Path

import pytest

from src.db import task_db
from src.workers import task_queue as task_queue_module
from src.workers.task_queue import TaskQueue, get_default_queue

MODULE_PATH = (
    pathlib.Path(__file__).resolve().parents[2] / "src" / "workers" / "task_queue.py"
)


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "queue.db"


@pytest.fixture()
def task_queue(db_path: Path) -> TaskQueue:
    """A queue bound to a temp database — impossible to build before the fix."""
    return TaskQueue(worker_id="test-worker", db_path=str(db_path))


class TestPathIsImported:
    """The missing import itself."""

    def test_module_defines_path(self) -> None:
        assert hasattr(task_queue_module, "Path")
        assert task_queue_module.Path is Path

    def test_path_is_imported_from_pathlib(self) -> None:
        tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
        imported = {
            alias.name
            for node in tree.body
            if isinstance(node, ast.ImportFrom) and node.module == "pathlib"
            for alias in node.names
        }
        assert "Path" in imported

    def test_every_global_the_constructor_touches_is_bound(self) -> None:
        """No name used in ``__init__`` is left undefined.

        ``from __future__ import annotations`` hides this class of bug from
        import-time checking — the constructor imported fine and only blew up
        when the ``db_path`` branch ran — so check the body explicitly rather
        than trusting a successful import.
        """
        import builtins

        tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
        klass = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "TaskQueue"
        )
        init = next(
            node
            for node in klass.body
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        )

        locals_and_params = {
            argument.arg for argument in init.args.args + init.args.kwonlyargs
        } | {
            node.id
            for node in ast.walk(init)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        }
        module_globals = set(vars(task_queue_module)) | set(vars(builtins))

        undefined = {
            node.id
            for node in ast.walk(init)
            if isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id not in locals_and_params
            and node.id not in module_globals
        }
        assert undefined == set()


class TestConstruction:
    """``TaskQueue(db_path=...)`` — the call that raised NameError."""

    def test_explicit_db_path_constructs(self, db_path: Path) -> None:
        TaskQueue(db_path=str(db_path))

    def test_the_database_file_is_created(self, db_path: Path) -> None:
        assert not db_path.exists()
        TaskQueue(db_path=str(db_path))
        assert db_path.exists()

    def test_the_schema_is_initialised(self, db_path: Path) -> None:
        TaskQueue(db_path=str(db_path))
        connection = sqlite3.connect(str(db_path))
        try:
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            }
        finally:
            connection.close()
        assert "task_jobs" in tables

    def test_a_missing_parent_directory_is_created(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b" / "queue.db"
        TaskQueue(db_path=str(nested))
        assert nested.exists()

    def test_db_path_is_stored_for_downstream_calls(self, db_path: Path) -> None:
        instance = TaskQueue(db_path=str(db_path))
        assert instance.db_path == str(db_path)

    def test_default_construction_still_works(self) -> None:
        """The ``db_path is None`` branch never touched ``Path``."""
        instance = TaskQueue()
        assert instance.db_path is None

    def test_constructor_defaults(self, db_path: Path) -> None:
        instance = TaskQueue(db_path=str(db_path))
        assert instance.is_running is False

    def test_worker_id_and_retry_settings_are_kept(self, db_path: Path) -> None:
        instance = TaskQueue(
            worker_id="w-7", max_retries=5, poll_interval=0.1, db_path=str(db_path)
        )
        assert instance._worker_id == "w-7"
        assert instance._max_retries == 5
        assert instance._poll_interval == 0.1


class TestLifecycle:
    """enqueue → dequeue → complete / fail / dead-letter, on a temp database."""

    def test_enqueue_returns_a_pending_job(self, task_queue: TaskQueue) -> None:
        job = task_queue.enqueue({"doc": "a.pdf"})
        assert job["status"] == "PENDING"
        assert job["id"]

    def test_enqueued_job_lands_in_the_given_database(
        self, task_queue: TaskQueue, db_path: Path
    ) -> None:
        job = task_queue.enqueue({"doc": "a.pdf"})
        connection = sqlite3.connect(str(db_path))
        try:
            row = connection.execute(
                "SELECT status FROM task_jobs WHERE id = ?", (job["id"],)
            ).fetchone()
        finally:
            connection.close()
        assert row == ("PENDING",)

    def test_payload_round_trips(self, task_queue: TaskQueue) -> None:
        payload = {"document_ids": ["a", "b"], "user_id": "teacher-1"}
        job = task_queue.enqueue(payload)
        stored = task_queue.get_job_status(job["id"])
        assert stored is not None
        assert dict(stored["payload"]) == payload

    def test_dequeue_claims_the_job_and_marks_it_processing(
        self, task_queue: TaskQueue
    ) -> None:
        task_queue.enqueue({"doc": "a.pdf"})
        claimed = task_queue.dequeue(timeout=1.0)
        assert claimed is not None
        assert claimed["status"] == "PROCESSING"

    def test_dequeue_records_the_worker_id(self, task_queue: TaskQueue) -> None:
        task_queue.enqueue({"doc": "a.pdf"})
        claimed = task_queue.dequeue(timeout=1.0)
        assert claimed is not None
        assert claimed["worker_id"] == "test-worker"

    def test_dequeue_on_an_empty_queue_returns_none(
        self, task_queue: TaskQueue
    ) -> None:
        assert task_queue.dequeue(timeout=0.05) is None

    def test_jobs_are_claimed_oldest_first(self, task_queue: TaskQueue) -> None:
        first = task_queue.enqueue({"n": 1})
        task_queue.enqueue({"n": 2})
        claimed = task_queue.dequeue(timeout=1.0)
        assert claimed is not None
        assert claimed["id"] == first["id"]

    def test_a_job_is_only_claimed_once(self, task_queue: TaskQueue) -> None:
        task_queue.enqueue({"doc": "a.pdf"})
        first = task_queue.dequeue(timeout=1.0)
        second = task_queue.dequeue(timeout=0.05)
        assert first is not None
        assert second is None

    def test_complete_stores_the_result(self, task_queue: TaskQueue) -> None:
        job = task_queue.enqueue({"doc": "a.pdf"})
        task_queue.dequeue(timeout=1.0)
        task_queue.complete(job["id"], {"score": 0.42})

        stored = task_queue.get_job_status(job["id"])
        assert stored is not None
        assert stored["status"] == "COMPLETED"
        assert dict(stored["result"]) == {"score": 0.42}

    def test_fail_requeues_while_retries_remain(self, db_path: Path) -> None:
        instance = TaskQueue(max_retries=3, db_path=str(db_path))
        job = instance.enqueue({"doc": "a.pdf"})
        instance.dequeue(timeout=1.0)
        instance.fail(job["id"], "transient parser error")

        stored = instance.get_job_status(job["id"])
        assert stored is not None
        assert stored["status"] == "PENDING"
        assert stored["retry_count"] == 1

    def test_fail_moves_to_dead_letter_once_retries_are_exhausted(
        self, db_path: Path
    ) -> None:
        instance = TaskQueue(max_retries=2, db_path=str(db_path))
        job = instance.enqueue({"doc": "a.pdf"})
        instance.fail(job["id"], "attempt 1")
        instance.fail(job["id"], "attempt 2")

        stored = instance.get_job_status(job["id"])
        assert stored is not None
        assert stored["status"] == "DEAD_LETTER"

    def test_dead_letter_bypasses_retries(self, task_queue: TaskQueue) -> None:
        job = task_queue.enqueue({"doc": "a.pdf"})
        task_queue.dead_letter(job["id"], "unsupported format")

        stored = task_queue.get_job_status(job["id"])
        assert stored is not None
        assert stored["status"] == "DEAD_LETTER"
        assert stored["retry_count"] == 0

    def test_get_job_status_for_an_unknown_id(self, task_queue: TaskQueue) -> None:
        assert task_queue.get_job_status("not-a-real-id") is None

    def test_submit_batch_scan_returns_a_job_id(self, task_queue: TaskQueue) -> None:
        job_id = task_queue.submit_batch_scan(["a.pdf", "b.pdf"], "teacher-1")
        stored = task_queue.get_job_status(job_id)
        assert stored is not None
        assert dict(stored["payload"])["type"] == "batch_scan"

    def test_submit_batch_scan_carries_priority(self, task_queue: TaskQueue) -> None:
        job_id = task_queue.submit_batch_scan(["a.pdf"], "teacher-1", priority=9)
        stored = task_queue.get_job_status(job_id)
        assert stored is not None
        assert dict(stored["payload"])["priority"] == 9

    def test_per_call_max_retries_overrides_the_instance_default(
        self, db_path: Path
    ) -> None:
        instance = TaskQueue(max_retries=3, db_path=str(db_path))
        job = instance.enqueue({"doc": "a.pdf"}, max_retries=1)
        stored = instance.get_job_status(job["id"])
        assert stored is not None
        assert stored["max_retries"] == 1


class TestIsolationBetweenDatabases:
    """The point of ``db_path``: two queues must not see each other's jobs."""

    def test_two_queues_do_not_share_jobs(self, tmp_path: Path) -> None:
        first = TaskQueue(db_path=str(tmp_path / "one.db"))
        second = TaskQueue(db_path=str(tmp_path / "two.db"))

        first.enqueue({"doc": "a.pdf"})
        assert second.dequeue(timeout=0.05) is None

    def test_each_queue_sees_only_its_own_rows(self, tmp_path: Path) -> None:
        first = TaskQueue(db_path=str(tmp_path / "one.db"))
        second = TaskQueue(db_path=str(tmp_path / "two.db"))

        first.enqueue({"doc": "a.pdf"})
        second.enqueue({"doc": "b.pdf"})
        second.enqueue({"doc": "c.pdf"})

        assert len(task_db.list_jobs(db_path=str(tmp_path / "one.db"))) == 1
        assert len(task_db.list_jobs(db_path=str(tmp_path / "two.db"))) == 2

    def test_state_survives_a_new_queue_object(self, db_path: Path) -> None:
        """The SQLite table, not the in-memory queue, is the source of truth."""
        first = TaskQueue(db_path=str(db_path))
        job = first.enqueue({"doc": "a.pdf"})

        second = TaskQueue(db_path=str(db_path))
        claimed = second.dequeue(timeout=1.0)
        assert claimed is not None
        assert claimed["id"] == job["id"]


class TestRequeueStaleProcessing:
    """Crash recovery, which also only works with a reachable database."""

    def test_returns_zero_when_nothing_is_processing(
        self, task_queue: TaskQueue
    ) -> None:
        assert task_queue.requeue_stale_processing() == 0

    def test_stale_processing_jobs_are_requeued(self, db_path: Path) -> None:
        instance = TaskQueue(max_retries=3, db_path=str(db_path))
        job = instance.enqueue({"doc": "a.pdf"})
        instance.dequeue(timeout=1.0)  # leaves it PROCESSING

        assert instance.requeue_stale_processing() == 1
        stored = instance.get_job_status(job["id"])
        assert stored is not None
        assert stored["status"] == "PENDING"

    def test_completed_jobs_are_left_alone(self, task_queue: TaskQueue) -> None:
        job = task_queue.enqueue({"doc": "a.pdf"})
        task_queue.dequeue(timeout=1.0)
        task_queue.complete(job["id"], {"ok": True})

        assert task_queue.requeue_stale_processing() == 0


class TestShutdownSignalling:
    def test_start_and_stop_toggle_is_running(self, task_queue: TaskQueue) -> None:
        assert task_queue.is_running is False
        task_queue.start()
        assert task_queue.is_running is True
        task_queue.stop()
        assert task_queue.is_running is False

    def test_signal_shutdown_makes_dequeue_return_none(
        self, task_queue: TaskQueue
    ) -> None:
        task_queue.signal_shutdown()
        assert task_queue.dequeue(timeout=1.0) is None

    def test_stop_also_signals_shutdown(self, task_queue: TaskQueue) -> None:
        task_queue.stop()
        assert task_queue._queue.get_nowait() is task_queue_module._SHUTDOWN

    def test_in_memory_queue_starts_empty(self, task_queue: TaskQueue) -> None:
        with pytest.raises(queue_module.Empty):
            task_queue._queue.get_nowait()


class TestDefaultQueueSingleton:
    def test_returns_the_same_instance(self) -> None:
        assert get_default_queue() is get_default_queue()

    def test_default_singleton_has_no_explicit_db_path(self) -> None:
        assert get_default_queue().db_path is None
