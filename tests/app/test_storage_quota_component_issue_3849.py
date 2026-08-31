"""
tests/app/test_storage_quota_component_issue_3849.py
-----------------------------------------------------
Tests for ``app/components/storage_quota.py`` (Issue #3849).

The component did not compile: the ``for`` loop inside
``get_total_corpus_storage_bytes()`` had been merged onto the same line as the
``if data_dir.exists():`` header that should contain it, so importing the
module raised ``SyntaxError`` and neither of its two public functions existed.

There was no test module for this component at all, which is why nothing in CI
noticed. These tests cover the byte accounting the broken loop was doing, the
percentage arithmetic and formatting in the renderer, and the Streamlit calls
the Settings tab depends on.
"""

from __future__ import annotations

import ast
import pathlib
from pathlib import Path
from unittest.mock import patch

import pytest

from app.components import storage_quota
from app.components.storage_quota import (
    STORAGE_LIMIT_GB,
    get_total_corpus_storage_bytes,
    render_storage_quota_progress,
)

MODULE_PATH = (
    pathlib.Path(__file__).resolve().parents[2]
    / "app"
    / "components"
    / "storage_quota.py"
)

GIB = 1024**3


@pytest.fixture()
def empty_usage():
    """Report zero bytes from ``calculate_storage_usage``.

    Isolates ``get_total_corpus_storage_bytes`` down to the directory walk,
    which is the part the syntax error destroyed.
    """
    with patch.object(
        storage_quota, "calculate_storage_usage", return_value={"total_bytes": 0}
    ):
        yield


@pytest.fixture()
def data_dir(tmp_path: Path):
    """Point the component's ``DATA_DIR`` at an empty temp directory."""
    directory = tmp_path / "data"
    directory.mkdir()
    with patch.object(storage_quota, "DATA_DIR", directory):
        yield directory


class TestModuleCompiles:
    """The syntax error itself."""

    def test_source_compiles(self) -> None:
        compile(MODULE_PATH.read_text(encoding="utf-8"), str(MODULE_PATH), "exec")

    def test_both_public_functions_exist(self) -> None:
        tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
        names = {
            node.name for node in tree.body if isinstance(node, ast.FunctionDef)
        }
        assert names == {
            "get_total_corpus_storage_bytes",
            "render_storage_quota_progress",
        }

    def test_the_loop_lives_inside_the_exists_guard(self) -> None:
        """The ``for`` is nested under ``if data_dir.exists():``, not beside it."""
        tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "get_total_corpus_storage_bytes"
        )
        guards = [node for node in function.body if isinstance(node, ast.If)]
        assert len(guards) == 1
        assert any(isinstance(child, ast.For) for child in guards[0].body)


class TestTotalCorpusStorageBytes:
    """The directory walk the collapsed line destroyed."""

    def test_starts_from_calculate_storage_usage(self, data_dir: Path) -> None:
        with patch.object(
            storage_quota,
            "calculate_storage_usage",
            return_value={"total_bytes": 4096},
        ):
            assert get_total_corpus_storage_bytes() == 4096

    def test_missing_total_bytes_key_defaults_to_zero(self, data_dir: Path) -> None:
        with patch.object(storage_quota, "calculate_storage_usage", return_value={}):
            assert get_total_corpus_storage_bytes() == 0

    def test_adds_a_plain_file_in_the_data_directory(
        self, empty_usage, data_dir: Path
    ) -> None:
        (data_dir / "notes.txt").write_bytes(b"x" * 100)
        assert get_total_corpus_storage_bytes() == 100

    def test_adds_several_files(self, empty_usage, data_dir: Path) -> None:
        for index in range(5):
            (data_dir / f"f{index}.txt").write_bytes(b"x" * 10)
        assert get_total_corpus_storage_bytes() == 50

    def test_walks_nested_directories(self, empty_usage, data_dir: Path) -> None:
        """The glob is ``**/*``, so nested files must be counted."""
        nested = data_dir / "a" / "b" / "c"
        nested.mkdir(parents=True)
        (nested / "deep.txt").write_bytes(b"x" * 64)
        assert get_total_corpus_storage_bytes() == 64

    def test_db_files_are_excluded(self, empty_usage, data_dir: Path) -> None:
        """``.db`` bytes already arrive via ``calculate_storage_usage``."""
        (data_dir / "corpus.db").write_bytes(b"x" * 999)
        assert get_total_corpus_storage_bytes() == 0

    def test_index_files_are_excluded(self, empty_usage, data_dir: Path) -> None:
        (data_dir / "corpus.index").write_bytes(b"x" * 999)
        assert get_total_corpus_storage_bytes() == 0

    def test_excluded_and_counted_files_together(
        self, empty_usage, data_dir: Path
    ) -> None:
        (data_dir / "corpus.db").write_bytes(b"x" * 500)
        (data_dir / "corpus.index").write_bytes(b"x" * 500)
        (data_dir / "upload.pdf").write_bytes(b"x" * 25)
        assert get_total_corpus_storage_bytes() == 25

    def test_directories_are_not_counted(self, empty_usage, data_dir: Path) -> None:
        (data_dir / "subdir").mkdir()
        assert get_total_corpus_storage_bytes() == 0

    def test_missing_data_directory_returns_the_base_total(
        self, tmp_path: Path
    ) -> None:
        with patch.object(storage_quota, "DATA_DIR", tmp_path / "absent"), patch.object(
            storage_quota,
            "calculate_storage_usage",
            return_value={"total_bytes": 7},
        ):
            assert get_total_corpus_storage_bytes() == 7

    def test_empty_data_directory_returns_the_base_total(self, data_dir: Path) -> None:
        with patch.object(
            storage_quota, "calculate_storage_usage", return_value={"total_bytes": 7}
        ):
            assert get_total_corpus_storage_bytes() == 7

    def test_broken_symlink_is_skipped(self, empty_usage, data_dir: Path) -> None:
        """``is_file()`` is False for a dangling link, so it never reaches stat."""
        (data_dir / "real.txt").write_bytes(b"x" * 10)
        (data_dir / "dangling.txt").symlink_to(data_dir / "gone.txt")
        assert get_total_corpus_storage_bytes() == 10

    def test_file_deleted_mid_walk_is_skipped(
        self, empty_usage, data_dir: Path
    ) -> None:
        """The ``except OSError`` guards the window between glob and stat."""
        (data_dir / "kept.txt").write_bytes(b"x" * 10)
        vanishing = data_dir / "vanishing.txt"
        vanishing.write_bytes(b"x" * 10)

        real_stat = Path.stat
        removed = {"done": False}

        def racing_stat(self, *args, **kwargs):
            if self.name == "vanishing.txt" and not removed["done"]:
                removed["done"] = True
                result = real_stat(self, *args, **kwargs)
                vanishing.unlink()
                return result
            return real_stat(self, *args, **kwargs)

        with patch.object(Path, "stat", racing_stat):
            total = get_total_corpus_storage_bytes()
        assert total == 10

    def test_result_is_an_int(self, empty_usage, data_dir: Path) -> None:
        (data_dir / "a.txt").write_bytes(b"x")
        assert isinstance(get_total_corpus_storage_bytes(), int)


class TestRenderStorageQuotaProgress:
    """The renderer's arithmetic and its Streamlit calls."""

    @staticmethod
    def _render(total_bytes: int, **kwargs):
        with patch.object(
            storage_quota, "get_total_corpus_storage_bytes", return_value=total_bytes
        ), patch("streamlit.markdown") as markdown, patch(
            "streamlit.progress"
        ) as progress, patch("streamlit.caption") as caption:
            result = render_storage_quota_progress(**kwargs)
        return result, markdown, progress, caption

    def test_returns_every_documented_key(self) -> None:
        result, _, _, _ = self._render(0)
        assert set(result) == {
            "total_bytes",
            "total_gb",
            "limit_gb",
            "percent",
            "caption",
        }

    def test_half_full(self) -> None:
        result, _, _, _ = self._render(5 * GIB, limit_gb=10.0)
        assert result["percent"] == pytest.approx(0.5)
        assert result["total_gb"] == pytest.approx(5.0)

    def test_empty_is_zero_percent(self) -> None:
        result, _, _, _ = self._render(0)
        assert result["percent"] == 0.0

    def test_percent_is_clamped_at_one_when_over_quota(self) -> None:
        result, _, _, _ = self._render(50 * GIB, limit_gb=10.0)
        assert result["percent"] == 1.0

    def test_exactly_at_the_limit_is_one(self) -> None:
        result, _, _, _ = self._render(10 * GIB, limit_gb=10.0)
        assert result["percent"] == pytest.approx(1.0)

    def test_custom_limit_is_honoured(self) -> None:
        result, _, _, _ = self._render(GIB, limit_gb=4.0)
        assert result["percent"] == pytest.approx(0.25)
        assert result["limit_gb"] == 4.0

    def test_default_limit_matches_the_module_constant(self) -> None:
        result, _, _, _ = self._render(0)
        assert result["limit_gb"] == STORAGE_LIMIT_GB

    def test_total_bytes_is_passed_through_unchanged(self) -> None:
        result, _, _, _ = self._render(123_456)
        assert result["total_bytes"] == 123_456

    def test_progress_receives_the_same_percent_it_returns(self) -> None:
        result, _, progress, _ = self._render(2 * GIB, limit_gb=8.0)
        progress.assert_called_once_with(result["percent"])

    def test_progress_argument_is_always_in_range(self) -> None:
        for total in (0, GIB, 10 * GIB, 100 * GIB):
            _, _, progress, _ = self._render(total, limit_gb=10.0)
            assert 0.0 <= progress.call_args.args[0] <= 1.0

    def test_heading_is_rendered(self) -> None:
        _, markdown, _, _ = self._render(0)
        markdown.assert_called_once_with("### 💾 Storage Quota Gauge")

    def test_caption_text(self) -> None:
        result, _, _, caption = self._render(5 * GIB, limit_gb=10.0)
        assert result["caption"] == "Storage Used: 5.0 GB / 10.0 GB (50%)"
        caption.assert_called_once_with(result["caption"])

    def test_caption_rounds_gigabytes_to_one_decimal(self) -> None:
        result, _, _, _ = self._render(int(1.25 * GIB), limit_gb=10.0)
        assert "1.2 GB" in result["caption"] or "1.3 GB" in result["caption"]

    def test_caption_percent_is_an_integer(self) -> None:
        result, _, _, _ = self._render(int(0.333 * GIB), limit_gb=1.0)
        assert "(33%)" in result["caption"]

    def test_streamlit_is_called_exactly_once_each(self) -> None:
        _, markdown, progress, caption = self._render(GIB)
        assert markdown.call_count == 1
        assert progress.call_count == 1
        assert caption.call_count == 1
