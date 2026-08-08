from unittest.mock import MagicMock, patch

import pytest

from src.core.document_parser import (
    _resolve_process_pool_workers,
    extract_texts,
    extract_texts_parallel,
)


class ImmediateFuture:
    def __init__(self, value):
        self._value = value

    def result(self):
        return self._value


class RecordingExecutor:
    recorded_max_workers = None

    def __init__(self, *, max_workers):
        type(self).recorded_max_workers = max_workers

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def submit(self, function, *args):
        return ImmediateFuture(function(*args))


def test_process_pool_is_capped_by_cpu_count(monkeypatch):
    files = {
        f"document-{index}.txt": f"text-{index}".encode()
        for index in range(10)
    }

    monkeypatch.setattr(
        "src.core.document_parser.os.cpu_count",
        lambda: 4,
    )
    monkeypatch.setattr(
        "src.core.document_parser._should_use_parallel",
        lambda: True,
    )
    monkeypatch.setattr(
        "src.core.document_parser._extract_single_file_helper",
        lambda data, name, language, dpi: data.decode(),
    )

    with patch(
        "concurrent.futures.ProcessPoolExecutor",
        RecordingExecutor,
    ):
        results, errors = extract_texts_parallel(
            files,
            max_workers=8,
        )

    assert RecordingExecutor.recorded_max_workers == 4
    assert list(results) == list(files)
    assert errors == {}


def test_process_pool_is_capped_by_requested_limit(monkeypatch):
    files = {
        f"document-{index}.txt": b"text"
        for index in range(8)
    }

    monkeypatch.setattr(
        "src.core.document_parser.os.cpu_count",
        lambda: 16,
    )
    monkeypatch.setattr(
        "src.core.document_parser._should_use_parallel",
        lambda: True,
    )
    monkeypatch.setattr(
        "src.core.document_parser._extract_single_file_helper",
        lambda *_args: "parsed",
    )

    with patch(
        "concurrent.futures.ProcessPoolExecutor",
        RecordingExecutor,
    ):
        extract_texts_parallel(
            files,
            max_workers=3,
        )

    assert RecordingExecutor.recorded_max_workers == 3


def test_process_pool_is_capped_by_file_count(monkeypatch):
    files = {
        "first.txt": b"first",
        "second.txt": b"second",
    }

    monkeypatch.setattr(
        "src.core.document_parser.os.cpu_count",
        lambda: 32,
    )
    monkeypatch.setattr(
        "src.core.document_parser._should_use_parallel",
        lambda: True,
    )
    monkeypatch.setattr(
        "src.core.document_parser._extract_single_file_helper",
        lambda data, *_args: data.decode(),
    )

    with patch(
        "concurrent.futures.ProcessPoolExecutor",
        RecordingExecutor,
    ):
        extract_texts_parallel(
            files,
            max_workers=20,
        )

    assert RecordingExecutor.recorded_max_workers == 2


def test_none_uses_available_cpu_count():
    with patch(
        "src.core.document_parser.os.cpu_count",
        return_value=6,
    ):
        assert _resolve_process_pool_workers(None, 20) == 6


def test_missing_cpu_count_falls_back_to_one():
    with patch(
        "src.core.document_parser.os.cpu_count",
        return_value=None,
    ):
        assert _resolve_process_pool_workers(None, 10) == 1
        assert _resolve_process_pool_workers(8, 10) == 1


@pytest.mark.parametrize("value", [0, -1])
def test_non_positive_max_workers_is_rejected(value):
    with pytest.raises(
        ValueError,
        match="at least 1",
    ):
        _resolve_process_pool_workers(value, 5)


@pytest.mark.parametrize(
    "value",
    [True, 1.5, "4", object()],
)
def test_non_integer_max_workers_is_rejected(value):
    with pytest.raises(
        TypeError,
        match="integer or None",
    ):
        _resolve_process_pool_workers(value, 5)


def test_one_worker_uses_sequential_path(monkeypatch):
    files = {
        "first.txt": b"first",
        "second.txt": b"second",
    }

    monkeypatch.setattr(
        "src.core.document_parser.os.cpu_count",
        lambda: 8,
    )
    monkeypatch.setattr(
        "src.core.document_parser._should_use_parallel",
        lambda: True,
    )
    helper = MagicMock(
        side_effect=lambda data, *_args: data.decode()
    )
    monkeypatch.setattr(
        "src.core.document_parser._extract_single_file_helper",
        helper,
    )

    with patch(
        "concurrent.futures.ProcessPoolExecutor"
    ) as executor:
        results, errors = extract_texts_parallel(
            files,
            max_workers=1,
        )

    executor.assert_not_called()
    assert results == {
        "first.txt": "first",
        "second.txt": "second",
    }
    assert errors == {}


def test_public_extract_texts_forwards_worker_limit(monkeypatch):
    uploaded = MagicMock()
    uploaded.name = "sample.txt"
    uploaded.read.return_value = b"sample"

    parallel = MagicMock(return_value=({"sample.txt": "ok"}, {}))
    monkeypatch.setattr(
        "src.core.document_parser.extract_texts_parallel",
        parallel,
    )

    result = extract_texts(
        [uploaded],
        session_id="session-1",
        max_workers=2,
    )

    assert result == {"sample.txt": "ok"}
    parallel.assert_called_once()
    assert parallel.call_args.kwargs == {
        "session_id": "session-1",
        "max_workers": 2,
    }
