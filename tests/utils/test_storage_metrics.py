"""Unit tests for src/utils/storage_metrics.py."""

import ast
import logging
import pathlib
from pathlib import Path
from unittest.mock import patch

import pytest

from src.utils.storage_metrics import (
    _deduplicate_paths,
    calculate_storage_usage,
    get_directory_size_bytes,
    get_faiss_index_paths,
    get_sqlite_db_paths,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "src" / "utils" / "storage_metrics.py"


def test_missing_db_files(tmp_path: Path) -> None:
    """Pass non-existent paths, assert sqlite_bytes == 0."""
    db_file = tmp_path / "nonexistent.db"
    usage = calculate_storage_usage(db_paths=[db_file], index_paths=[])
    assert usage["sqlite_bytes"] == 0


def test_missing_index_files(tmp_path: Path) -> None:
    """Pass non-existent paths, assert faiss_bytes == 0."""
    index_file = tmp_path / "nonexistent.index"
    usage = calculate_storage_usage(db_paths=[], index_paths=[index_file])
    assert usage["faiss_bytes"] == 0


def test_real_file(tmp_path: Path) -> None:
    """Create a temp file and verify its size is reported correctly."""
    db_file = tmp_path / "test_corpus.db"
    db_file.write_bytes(b"0" * 1024)
    usage = calculate_storage_usage(db_paths=[db_file], index_paths=[])
    assert usage["sqlite_bytes"] == 1024


def test_get_sqlite_db_paths() -> None:
    """Test get_sqlite_db_paths returns a list of Path objects."""
    paths = get_sqlite_db_paths()
    assert isinstance(paths, list)
    for p in paths:
        assert isinstance(p, Path)


def test_get_faiss_index_paths() -> None:
    """Test get_faiss_index_paths returns a list of Path objects."""
    paths = get_faiss_index_paths()
    assert isinstance(paths, list)
    for p in paths:
        assert isinstance(p, Path)


def test_path_resolution_logs_debug_warning(caplog) -> None:
    """Verify that exceptions during path resolution log debug warnings."""
    with caplog.at_level(logging.DEBUG):
        with patch(
            "src.db.corpus_db.get_corpus_db_path",
            side_effect=Exception("Database path resolution error"),
        ):
            get_sqlite_db_paths()
            assert (
                "Could not resolve path: Database path resolution error" in caplog.text
            )


class TestCalculateStorageUsageFileCounts:
    """Test suite for file count tracking in calculate_storage_usage() (Issue #2253)."""

    def test_returns_zero_counts_for_empty_paths(self, tmp_path):
        """Verify file counts are 0 when no files exist at provided paths."""
        from src.utils.storage_metrics import calculate_storage_usage

        # Pass empty lists to simulate no files found
        result = calculate_storage_usage(db_paths=[], index_paths=[])

        assert result["sqlite_file_count"] == 0
        assert result["faiss_file_count"] == 0
        assert result["formatted_total"] == "0.00 MB"

    def test_counts_sqlite_files_correctly(self, tmp_path):
        """Verify sqlite_file_count increments for each valid .db file."""
        from src.utils.storage_metrics import calculate_storage_usage

        # Create 3 dummy SQLite files
        db_paths = []
        for i in range(3):
            db_file = tmp_path / f"test_{i}.db"
            db_file.write_bytes(b"x" * 1024)  # 1KB each
            db_paths.append(db_file)

        result = calculate_storage_usage(db_paths=db_paths, index_paths=[])

        assert result["sqlite_file_count"] == 3
        assert result["faiss_file_count"] == 0
        assert result["sqlite_bytes"] == 3072

    def test_counts_faiss_files_correctly(self, tmp_path):
        """Verify faiss_file_count increments for each valid .index file."""
        from src.utils.storage_metrics import calculate_storage_usage

        # Create 2 dummy FAISS index files
        index_paths = []
        for i in range(2):
            idx_file = tmp_path / f"corpus_{i}.index"
            idx_file.write_bytes(b"y" * 2048)  # 2KB each
            index_paths.append(idx_file)

        result = calculate_storage_usage(db_paths=[], index_paths=index_paths)

        assert result["sqlite_file_count"] == 0
        assert result["faiss_file_count"] == 2
        assert result["faiss_bytes"] == 4096

    def test_ignores_nonexistent_paths_in_count(self, tmp_path):
        """Verify nonexistent paths don't increment the file count."""
        from src.utils.storage_metrics import calculate_storage_usage

        existing_db = tmp_path / "real.db"
        existing_db.write_bytes(b"data")

        nonexistent_db = tmp_path / "missing.db"

        result = calculate_storage_usage(
            db_paths=[existing_db, nonexistent_db], index_paths=[]
        )

        # Should only count the existing file
        assert result["sqlite_file_count"] == 1

    def test_ignores_directories_in_count(self, tmp_path):
        """Verify directories ending in .db don't increment the file count."""
        from src.utils.storage_metrics import calculate_storage_usage

        # Create a directory with .db extension
        db_dir = tmp_path / "fake.db"
        db_dir.mkdir()

        result = calculate_storage_usage(db_paths=[db_dir], index_paths=[])

        assert result["sqlite_file_count"] == 0


class TestModuleParses:
    """Guards for the breakage in Issue #2555.

    Both public helpers had their ``def`` line and docstring collapsed onto a
    single line, and ``get_faiss_index_paths`` additionally had its ``return``
    dedented to column 0. The module did not compile, so every test in this
    file errored at collection and three ``app/`` modules failed to import.
    """

    def test_source_compiles(self):
        """The assertion that Issue #2555 failed."""
        compile(
            MODULE_PATH.read_text(encoding="utf-8"),
            "src/utils/storage_metrics.py",
            "exec",
        )

    @pytest.mark.parametrize(
        "name",
        [
            "_deduplicate_paths",
            "get_sqlite_db_paths",
            "get_faiss_index_paths",
            "calculate_storage_usage",
        ],
    )
    def test_function_is_defined_exactly_once(self, name):
        """A second definition would silently shadow the first."""
        tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
        matches = [
            node.lineno
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        ]
        assert len(matches) == 1, f"{name} defined at lines {matches}"

    def test_public_helpers_have_docstrings(self):
        """The collapsed lines were docstrings; make sure they came back."""
        for func in (get_sqlite_db_paths, get_faiss_index_paths):
            assert func.__doc__, f"{func.__name__} has no docstring"

    def test_every_function_body_is_indented(self):
        """Catch a dedented statement leaking back to module scope.

        The stray ``return _deduplicate_paths(paths)`` at column 0 is what made
        this module unparseable. ``ast`` cannot see indentation, so compare each
        function's declared end against the next top-level node instead: any
        statement that belongs to a function but sits at module level shows up
        as an extra top-level node between the two.
        """
        tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, (ast.Return, ast.Pass)):
                pytest.fail(
                    f"line {node.lineno}: {type(node).__name__} at module scope - "
                    "a function body statement is dedented"
                )


class TestDeduplicatePaths:
    """Direct coverage for the helper both public functions delegate to."""

    def test_removes_paths_that_resolve_to_the_same_file(self, tmp_path):
        db_file = tmp_path / "corpus.db"
        db_file.write_bytes(b"x")

        indirect = tmp_path / "sub" / ".." / "corpus.db"
        (tmp_path / "sub").mkdir()

        result = _deduplicate_paths([db_file, indirect, db_file])

        assert len(result) == 1

    def test_keeps_the_first_occurrence(self, tmp_path):
        """Order matters: callers show these paths to the user as-is."""
        first = tmp_path / "a.db"
        first.write_bytes(b"x")
        duplicate = tmp_path / "." / "a.db"

        result = _deduplicate_paths([first, duplicate])

        assert result == [first]

    def test_keeps_genuinely_distinct_paths(self, tmp_path):
        """``data/corpus.db`` and a root ``corpus.db`` are two real files."""
        root_db = tmp_path / "corpus.db"
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        nested_db = data_dir / "corpus.db"
        root_db.write_bytes(b"x")
        nested_db.write_bytes(b"y")

        result = _deduplicate_paths([root_db, nested_db])

        assert len(result) == 2

    def test_handles_paths_that_do_not_exist(self, tmp_path):
        """``resolve()`` is non-strict, so missing files still de-duplicate."""
        missing = tmp_path / "never_created.db"

        result = _deduplicate_paths([missing, missing])

        assert result == [missing]

    def test_empty_input_returns_empty_list(self):
        assert _deduplicate_paths([]) == []

    def test_unresolvable_path_is_skipped_not_raised(self, caplog):
        """A path that cannot be resolved must not take the whole call down."""
        broken = Path("broken.db")

        with caplog.at_level(logging.DEBUG):
            with patch.object(
                Path, "resolve", side_effect=OSError("resolution blew up")
            ):
                result = _deduplicate_paths([broken])

        assert result == []
        assert "resolution blew up" in caplog.text


class TestGetSqliteDbPaths:
    """Coverage for the SQLite discovery helper (Issue #2555)."""

    def test_returns_paths(self):
        paths = get_sqlite_db_paths()

        assert isinstance(paths, list)
        assert all(isinstance(p, Path) for p in paths)

    def test_result_contains_no_duplicate_resolved_paths(self):
        """The corpus DB is discovered twice: configured, then by glob."""
        paths = get_sqlite_db_paths()
        resolved = [p.resolve() for p in paths]

        assert len(resolved) == len(set(resolved)), (
            f"duplicate database paths reported: {sorted(map(str, resolved))}"
        )

    def test_survives_every_configured_lookup_failing(self, caplog):
        """A partially installed environment must still return glob results."""
        with caplog.at_level(logging.DEBUG):
            with (
                patch(
                    "src.db.corpus_db.get_corpus_db_path",
                    side_effect=Exception("corpus unavailable"),
                ),
                patch(
                    "src.db.auth.get_auth_db_path",
                    side_effect=Exception("auth unavailable"),
                ),
            ):
                paths = get_sqlite_db_paths()

        assert isinstance(paths, list)
        assert "corpus unavailable" in caplog.text
        assert "auth unavailable" in caplog.text


class TestGetFaissIndexPaths:
    """Coverage for the FAISS discovery helper (Issue #2555)."""

    def test_returns_paths(self):
        paths = get_faiss_index_paths()

        assert isinstance(paths, list)
        assert all(isinstance(p, Path) for p in paths)

    def test_always_offers_the_default_index_locations(self):
        """Reported before the index exists, so the UI can show 0 bytes."""
        names = {p.name for p in get_faiss_index_paths()}

        assert "corpus.index" in names

    def test_result_contains_no_duplicate_resolved_paths(self):
        """The defaults are appended and then globbed again."""
        paths = get_faiss_index_paths()
        resolved = [p.resolve() for p in paths]

        assert len(resolved) == len(set(resolved)), (
            f"duplicate index paths reported: {sorted(map(str, resolved))}"
        )

    def test_only_index_files_are_reported(self):
        offenders = [p for p in get_faiss_index_paths() if p.suffix != ".index"]

        assert not offenders, f"non-index paths reported: {offenders}"


class TestCalculateStorageUsageContract:
    """The return shape three UI call sites read keys out of."""

    EXPECTED_KEYS = {
        "sqlite_bytes",
        "faiss_bytes",
        "total_bytes",
        "sqlite_mb",
        "faiss_mb",
        "total_mb",
        "formatted_total",
        "formatted_sqlite",
        "formatted_faiss",
        "sqlite_file_count",
        "faiss_file_count",
    }

    def test_returns_every_documented_key(self, tmp_path):
        result = calculate_storage_usage(db_paths=[], index_paths=[])

        assert set(result) == self.EXPECTED_KEYS

    def test_totals_are_the_sum_of_the_parts(self, tmp_path):
        db_file = tmp_path / "corpus.db"
        db_file.write_bytes(b"a" * 3000)
        index_file = tmp_path / "corpus.index"
        index_file.write_bytes(b"b" * 5000)

        result = calculate_storage_usage(db_paths=[db_file], index_paths=[index_file])

        assert result["sqlite_bytes"] == 3000
        assert result["faiss_bytes"] == 5000
        assert result["total_bytes"] == 8000

    def test_megabytes_are_rounded_to_two_places(self, tmp_path):
        db_file = tmp_path / "corpus.db"
        db_file.write_bytes(b"a" * (1024 * 1024 + 1024))  # 1.0009765625 MB

        result = calculate_storage_usage(db_paths=[db_file], index_paths=[])

        assert result["sqlite_mb"] == 1.0
        assert result["formatted_sqlite"] == "1.00 MB"

    def test_formatted_strings_always_carry_two_decimals_and_a_unit(self):
        result = calculate_storage_usage(db_paths=[], index_paths=[])

        for key in ("formatted_total", "formatted_sqlite", "formatted_faiss"):
            assert result[key].endswith(" MB")
            assert result[key].split(" ")[0].split(".")[1] == "00"

    def test_falls_back_to_discovery_when_no_paths_are_passed(self):
        """The no-argument call is what the Streamlit widgets actually make."""
        with (
            patch(
                "src.utils.storage_metrics.get_sqlite_db_paths", return_value=[]
            ) as mock_db,
            patch(
                "src.utils.storage_metrics.get_faiss_index_paths", return_value=[]
            ) as mock_index,
        ):
            result = calculate_storage_usage()

        mock_db.assert_called_once_with()
        mock_index.assert_called_once_with()
        assert result["total_bytes"] == 0

    def test_unreadable_file_is_skipped_not_raised(self, tmp_path, caplog):
        """``stat()`` can fail on a path we can see but cannot read."""
        db_file = tmp_path / "corpus.db"
        db_file.write_bytes(b"x" * 100)

        with caplog.at_level(logging.DEBUG):
            with patch.object(Path, "stat", side_effect=OSError("permission denied")):
                result = calculate_storage_usage(db_paths=[db_file], index_paths=[])

        assert result["sqlite_bytes"] == 0
        assert result["sqlite_file_count"] == 0
        assert "permission denied" in caplog.text

    def test_index_directory_is_not_counted(self, tmp_path):
        """Mirror of the existing .db directory test, for the FAISS side."""
        index_dir = tmp_path / "fake.index"
        index_dir.mkdir()

        result = calculate_storage_usage(db_paths=[], index_paths=[index_dir])

        assert result["faiss_file_count"] == 0
        assert result["faiss_bytes"] == 0
