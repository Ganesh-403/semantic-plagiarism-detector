"""
tests/utils/test_storage_metrics_db_discovery_issue_3848.py
------------------------------------------------------------
Regression tests for Issue #3848.

``src/utils/storage_metrics.py`` did not compile: inside
``get_sqlite_db_paths()`` the ``data_dir = DATA_DIR`` assignment and the
``for folder in [base_dir, data_dir]:`` loop that follows it had been
collapsed onto one line, so the whole module raised ``SyntaxError`` at import
and every storage feature that imports it went down with it.

``tests/utils/test_storage_metrics.py`` already covers the arithmetic in
``calculate_storage_usage`` and the de-duplication helper. What it does not
cover is the loop body that the collapsed line destroyed — nothing asserted
that ``get_sqlite_db_paths()`` actually *discovers* ``*.db`` files under the
two search roots. These tests close that gap, and pick up the three functions
in the module that had no coverage at all:
``calculate_database_fragmentation``, ``get_storage_by_class`` and the
``get_projected_days_until_full`` boundary cases.
"""

from __future__ import annotations

import ast
import pathlib
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from src.utils.storage_metrics import (
    _connect_storage_history,
    calculate_database_fragmentation,
    get_projected_days_until_full,
    get_sqlite_db_paths,
    get_storage_by_class,
    record_storage_snapshot,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "src" / "utils" / "storage_metrics.py"


def _make_db(path: Path, *, pages: int = 1) -> Path:
    """Create a small but real SQLite file at ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path))
    try:
        connection.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, blob TEXT)")
        connection.executemany(
            "INSERT INTO t (blob) VALUES (?)",
            [("x" * 512,) for _ in range(pages * 8)],
        )
        connection.commit()
    finally:
        connection.close()
    return path


def _make_corpus_db(path: Path, rows: list[tuple[str, str | None, int]]) -> Path:
    """Create a corpus-shaped database with ``documents`` and ``chunks``.

    ``rows`` is a list of ``(filename, class_section, is_deleted)`` tuples.
    Each document gets one chunk so the aggregate columns are predictable.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path))
    try:
        connection.execute(
            """
            CREATE TABLE documents (
                filename TEXT PRIMARY KEY,
                class_section TEXT,
                is_deleted INTEGER
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE chunks (
                vector_id INTEGER PRIMARY KEY,
                filename TEXT,
                chunk_text TEXT,
                embedding BLOB
            )
            """
        )
        for index, (filename, class_section, is_deleted) in enumerate(rows):
            connection.execute(
                "INSERT INTO documents (filename, class_section, is_deleted)"
                " VALUES (?, ?, ?)",
                (filename, class_section, is_deleted),
            )
            connection.execute(
                "INSERT INTO chunks (vector_id, filename, chunk_text, embedding)"
                " VALUES (?, ?, ?, ?)",
                (index + 1, filename, "abcde", b"1234"),
            )
        connection.commit()
    finally:
        connection.close()
    return path


class TestModuleCompiles:
    """The syntax error itself."""

    def test_source_compiles(self) -> None:
        compile(MODULE_PATH.read_text(encoding="utf-8"), str(MODULE_PATH), "exec")

    def test_no_statement_follows_an_assignment_on_the_same_line(self) -> None:
        """No line pairs an assignment with a trailing compound keyword.

        This is the exact shape that broke the module:
        ``data_dir = DATA_DIR    for folder in ...``.
        """
        offenders: list[tuple[int, str]] = []
        for number, line in enumerate(
            MODULE_PATH.read_text(encoding="utf-8").splitlines(), start=1
        ):
            stripped = line.strip()
            if "=" not in stripped or stripped.startswith("#"):
                continue
            _, _, tail = stripped.partition("=")
            for keyword in (" for ", " if ", " while ", " with ", " return "):
                if keyword in f" {tail} ":
                    try:
                        ast.parse(stripped)
                    except SyntaxError:
                        offenders.append((number, line))
                    break
        assert offenders == []

    def test_both_search_root_loops_have_the_same_shape(self) -> None:
        """``get_sqlite_db_paths`` and ``get_faiss_index_paths`` still agree.

        The FAISS variant was the intact copy used to reconstruct the broken
        one; if they drift apart again that is worth knowing.
        """
        tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
        loop_targets: dict[str, list[str]] = {}
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            if node.name not in {"get_sqlite_db_paths", "get_faiss_index_paths"}:
                continue
            loop_targets[node.name] = [
                child.target.id
                for child in ast.walk(node)
                if isinstance(child, ast.For) and isinstance(child.target, ast.Name)
            ]
        assert loop_targets["get_sqlite_db_paths"] == ["folder", "file_path"]
        assert loop_targets["get_faiss_index_paths"] == ["folder", "file_path"]


class TestSqliteDbDiscovery:
    """The loop body that the collapsed line destroyed."""

    @pytest.fixture()
    def isolated_roots(self, tmp_path: Path):
        """Redirect ``DATA_DIR`` at an empty temp directory.

        ``base_dir`` is derived from ``__file__`` inside the function and
        stays pointed at the repository root, so the tests below assert on
        membership rather than exact equality — the repository's own ``*.db``
        files are legitimately discovered too and must not make these
        assertions brittle.
        """
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        with patch("src.utils.storage_metrics.DATA_DIR", data_dir):
            yield data_dir

    def test_finds_a_db_file_in_the_data_directory(self, isolated_roots: Path) -> None:
        created = _make_db(isolated_roots / "discovered.db")
        found = {path.resolve() for path in get_sqlite_db_paths()}
        assert created.resolve() in found

    def test_finds_several_db_files(self, isolated_roots: Path) -> None:
        created = {
            _make_db(isolated_roots / f"corpus_{index}.db").resolve()
            for index in range(4)
        }
        found = {path.resolve() for path in get_sqlite_db_paths()}
        assert created <= found

    def test_ignores_non_db_suffixes(self, isolated_roots: Path) -> None:
        (isolated_roots / "notes.txt").write_text("hello", encoding="utf-8")
        (isolated_roots / "corpus.index").write_bytes(b"\x00\x01")
        (isolated_roots / "archive.sqlite").write_bytes(b"\x00")
        found = {path.name for path in get_sqlite_db_paths()}
        assert "notes.txt" not in found
        assert "corpus.index" not in found
        assert "archive.sqlite" not in found

    def test_glob_is_not_recursive(self, isolated_roots: Path) -> None:
        """``glob("*.db")`` is deliberately shallow — nested files are skipped."""
        nested = _make_db(isolated_roots / "nested" / "deep.db")
        found = {path.resolve() for path in get_sqlite_db_paths()}
        assert nested.resolve() not in found

    def test_missing_data_directory_is_tolerated(self, tmp_path: Path) -> None:
        """The loop guards each root with ``folder.exists()``."""
        with patch("src.utils.storage_metrics.DATA_DIR", tmp_path / "absent"):
            assert isinstance(get_sqlite_db_paths(), list)

    def test_returns_path_objects(self, isolated_roots: Path) -> None:
        _make_db(isolated_roots / "corpus.db")
        assert all(isinstance(path, Path) for path in get_sqlite_db_paths())

    def test_discovered_file_is_deduplicated(self, isolated_roots: Path) -> None:
        created = _make_db(isolated_roots / "corpus.db")
        resolved = [path.resolve() for path in get_sqlite_db_paths()]
        assert resolved.count(created.resolve()) == 1


class TestDatabaseFragmentation:
    """``calculate_database_fragmentation`` had no coverage."""

    def test_reports_page_and_freelist_counts(self, tmp_path: Path) -> None:
        db_file = _make_db(tmp_path / "frag.db", pages=4)
        report = calculate_database_fragmentation(str(db_file))
        assert report["page_count"] > 0
        assert report["freelist_count"] >= 0
        assert 0.0 <= report["fragmentation_percentage"] <= 100.0

    def test_freshly_written_database_is_optimal(self, tmp_path: Path) -> None:
        db_file = _make_db(tmp_path / "fresh.db", pages=4)
        report = calculate_database_fragmentation(str(db_file))
        assert report["status"] == "OPTIMAL"

    def test_empty_database_is_reported_not_divided_by_zero(
        self, tmp_path: Path
    ) -> None:
        db_file = tmp_path / "empty.db"
        sqlite3.connect(str(db_file)).close()
        report = calculate_database_fragmentation(str(db_file))
        assert report == {
            "freelist_count": 0,
            "page_count": 0,
            "fragmentation_percentage": 0.0,
            "status": "EMPTY_DATABASE",
        }

    def test_heavy_deletion_recommends_vacuum(self, tmp_path: Path) -> None:
        db_file = tmp_path / "bloated.db"
        connection = sqlite3.connect(str(db_file))
        try:
            connection.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, blob TEXT)")
            connection.executemany(
                "INSERT INTO t (blob) VALUES (?)",
                [("x" * 4096,) for _ in range(400)],
            )
            connection.commit()
            connection.execute("DELETE FROM t WHERE id > 20")
            connection.commit()
        finally:
            connection.close()

        report = calculate_database_fragmentation(str(db_file))
        assert report["fragmentation_percentage"] > 20.0
        assert report["status"] == "VACUUM_RECOMMENDED"

    def test_unreadable_file_returns_error_dict(self, tmp_path: Path) -> None:
        not_a_db = tmp_path / "garbage.db"
        not_a_db.write_bytes(b"this is definitely not a sqlite header")
        report = calculate_database_fragmentation(str(not_a_db))
        assert report["error"] == "SQLITE_QUERY_FAILURE"
        assert report["fragmentation_percentage"] == -1.0

    def test_error_dict_is_returned_not_raised(self, tmp_path: Path) -> None:
        missing_dir = tmp_path / "no" / "such" / "dir" / "x.db"
        report = calculate_database_fragmentation(str(missing_dir))
        assert report["fragmentation_percentage"] == -1.0


class TestStorageByClass:
    """``get_storage_by_class`` had no coverage."""

    def test_returns_empty_when_corpus_db_absent(self, tmp_path: Path) -> None:
        with patch(
            "src.db.corpus_db.get_corpus_db_path",
            return_value=tmp_path / "missing.db",
        ):
            assert get_storage_by_class() == []

    def test_groups_documents_by_class_section(self, tmp_path: Path) -> None:
        db_file = _make_corpus_db(
            tmp_path / "corpus.db",
            [
                ("a.pdf", "CS101", 0),
                ("b.pdf", "CS101", 0),
                ("c.pdf", "CS102", 0),
            ],
        )
        with patch("src.db.corpus_db.get_corpus_db_path", return_value=db_file):
            rows = get_storage_by_class()

        by_section = {row["class_section"]: row for row in rows}
        assert by_section["CS101"]["document_count"] == 2
        assert by_section["CS102"]["document_count"] == 1

    def test_blank_section_is_labelled_unassigned(self, tmp_path: Path) -> None:
        db_file = _make_corpus_db(
            tmp_path / "corpus.db",
            [("a.pdf", "", 0), ("b.pdf", "", 0)],
        )
        with patch("src.db.corpus_db.get_corpus_db_path", return_value=db_file):
            rows = get_storage_by_class()

        assert [row["class_section"] for row in rows] == ["Unassigned"]
        assert rows[0]["document_count"] == 2

    def test_null_section_is_labelled_unassigned(self, tmp_path: Path) -> None:
        db_file = _make_corpus_db(
            tmp_path / "corpus.db",
            [("a.pdf", None, 0), ("b.pdf", None, 0)],
        )
        with patch("src.db.corpus_db.get_corpus_db_path", return_value=db_file):
            rows = get_storage_by_class()

        assert [row["class_section"] for row in rows] == ["Unassigned"]
        assert rows[0]["document_count"] == 2

    def test_deleted_documents_are_excluded(self, tmp_path: Path) -> None:
        db_file = _make_corpus_db(
            tmp_path / "corpus.db",
            [("live.pdf", "CS101", 0), ("gone.pdf", "CS101", 1)],
        )
        with patch("src.db.corpus_db.get_corpus_db_path", return_value=db_file):
            rows = get_storage_by_class()

        assert rows[0]["document_count"] == 1

    def test_every_documented_key_is_present_and_typed(self, tmp_path: Path) -> None:
        db_file = _make_corpus_db(tmp_path / "corpus.db", [("a.pdf", "CS101", 0)])
        with patch("src.db.corpus_db.get_corpus_db_path", return_value=db_file):
            rows = get_storage_by_class()

        assert set(rows[0]) == {
            "class_section",
            "document_count",
            "chunk_count",
            "estimated_bytes",
        }
        assert isinstance(rows[0]["class_section"], str)
        assert isinstance(rows[0]["document_count"], int)
        assert isinstance(rows[0]["chunk_count"], int)
        assert isinstance(rows[0]["estimated_bytes"], int)

    def test_estimated_bytes_sums_text_and_embedding_lengths(
        self, tmp_path: Path
    ) -> None:
        """Each chunk contributes ``len(chunk_text) + len(embedding)``."""
        db_file = _make_corpus_db(
            tmp_path / "corpus.db",
            [("a.pdf", "CS101", 0), ("b.pdf", "CS101", 0)],
        )
        with patch("src.db.corpus_db.get_corpus_db_path", return_value=db_file):
            rows = get_storage_by_class()

        assert rows[0]["estimated_bytes"] == 2 * (len("abcde") + len(b"1234"))

    def test_rows_are_ordered_by_estimated_bytes_descending(
        self, tmp_path: Path
    ) -> None:
        db_file = _make_corpus_db(
            tmp_path / "corpus.db",
            [("a.pdf", "SMALL", 0), ("b.pdf", "BIG", 0), ("c.pdf", "BIG", 0)],
        )
        with patch("src.db.corpus_db.get_corpus_db_path", return_value=db_file):
            rows = get_storage_by_class()

        sizes = [row["estimated_bytes"] for row in rows]
        assert sizes == sorted(sizes, reverse=True)

    def test_malformed_corpus_db_returns_empty_list(self, tmp_path: Path) -> None:
        broken = tmp_path / "corpus.db"
        broken.write_bytes(b"not sqlite")
        with patch("src.db.corpus_db.get_corpus_db_path", return_value=broken):
            assert get_storage_by_class() == []


class TestProjectionBoundaries:
    """``get_projected_days_until_full`` edges the existing suite does not reach."""

    @staticmethod
    def _write_history(db_path: Path, rows: list[tuple[str, int, int]]) -> None:
        connection = _connect_storage_history(db_path)
        try:
            connection.executemany(
                "INSERT INTO storage_history (date, db_size_bytes, temp_size_bytes)"
                " VALUES (?, ?, ?)",
                rows,
            )
            connection.commit()
        finally:
            connection.close()

    def test_already_at_the_limit_returns_zero(self, tmp_path: Path) -> None:
        db_path = tmp_path / "history.db"
        self._write_history(
            db_path, [("2026-01-01", 100, 0), ("2026-01-11", 1_000, 0)]
        )
        assert get_projected_days_until_full(1_000, db_path=db_path) == 0.0

    def test_over_the_limit_returns_zero(self, tmp_path: Path) -> None:
        db_path = tmp_path / "history.db"
        self._write_history(
            db_path, [("2026-01-01", 100, 0), ("2026-01-11", 5_000, 0)]
        )
        assert get_projected_days_until_full(1_000, db_path=db_path) == 0.0

    def test_flat_usage_is_never_projected_to_fill(self, tmp_path: Path) -> None:
        db_path = tmp_path / "history.db"
        self._write_history(
            db_path, [("2026-01-01", 500, 0), ("2026-01-11", 500, 0)]
        )
        assert get_projected_days_until_full(1_000, db_path=db_path) == float("inf")

    def test_shrinking_usage_is_never_projected_to_fill(self, tmp_path: Path) -> None:
        db_path = tmp_path / "history.db"
        self._write_history(
            db_path, [("2026-01-01", 900, 0), ("2026-01-11", 400, 0)]
        )
        assert get_projected_days_until_full(1_000, db_path=db_path) == float("inf")

    def test_two_snapshots_on_the_same_day_cannot_be_projected(
        self, tmp_path: Path
    ) -> None:
        """Zero elapsed days would divide by zero, so the guard returns inf."""
        db_path = tmp_path / "history.db"
        connection = _connect_storage_history(db_path)
        try:
            connection.execute(
                "INSERT INTO storage_history VALUES ('2026-01-01', 100, 0)"
            )
            connection.execute(
                "INSERT OR REPLACE INTO storage_history"
                " VALUES ('2026-01-01', 900, 0)"
            )
            connection.commit()
        finally:
            connection.close()
        assert get_projected_days_until_full(1_000, db_path=db_path) == float("inf")

    def test_temp_bytes_count_toward_the_projection(self, tmp_path: Path) -> None:
        db_path = tmp_path / "history.db"
        self._write_history(
            db_path, [("2026-01-01", 0, 100), ("2026-01-11", 0, 600)]
        )
        # 500 bytes over 10 days = 50/day; 1000 - 600 = 400 remaining.
        assert get_projected_days_until_full(1_000, db_path=db_path) == pytest.approx(
            8.0
        )

    def test_null_columns_are_treated_as_zero(self, tmp_path: Path) -> None:
        db_path = tmp_path / "history.db"
        connection = _connect_storage_history(db_path)
        try:
            connection.execute(
                "INSERT INTO storage_history VALUES ('2026-01-01', NULL, NULL)"
            )
            connection.execute(
                "INSERT INTO storage_history VALUES ('2026-01-11', 500, NULL)"
            )
            connection.commit()
        finally:
            connection.close()
        assert get_projected_days_until_full(1_000, db_path=db_path) == pytest.approx(
            10.0
        )


class TestSnapshotRoundTrip:
    """``record_storage_snapshot`` feeding ``get_projected_days_until_full``."""

    def test_snapshot_is_upserted_not_duplicated(self, tmp_path: Path) -> None:
        db_path = tmp_path / "history.db"
        record_storage_snapshot(db_path=db_path)
        record_storage_snapshot(db_path=db_path)

        connection = _connect_storage_history(db_path)
        try:
            count = connection.execute(
                "SELECT COUNT(*) FROM storage_history"
            ).fetchone()[0]
        finally:
            connection.close()
        assert count == 1

    def test_single_snapshot_cannot_be_projected(self, tmp_path: Path) -> None:
        db_path = tmp_path / "history.db"
        record_storage_snapshot(db_path=db_path)
        assert get_projected_days_until_full(
            10**12, db_path=db_path
        ) == float("inf")

    def test_history_table_has_the_expected_columns(self, tmp_path: Path) -> None:
        connection = _connect_storage_history(tmp_path / "history.db")
        try:
            columns = [
                row[1]
                for row in connection.execute(
                    "PRAGMA table_info(storage_history)"
                ).fetchall()
            ]
        finally:
            connection.close()
        assert columns == ["date", "db_size_bytes", "temp_size_bytes"]
