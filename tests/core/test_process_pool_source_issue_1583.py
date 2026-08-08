from pathlib import Path


SOURCE = Path("src/core/document_parser.py")
TESTS = Path(
    "tests/core/test_document_parser_process_pool_issue_1583.py"
)


def test_bulk_helper_accepts_worker_limit():
    source = SOURCE.read_text(encoding="utf-8")

    assert "def extract_texts_parallel(" in source
    assert "max_workers: int | None = None" in source


def test_executor_uses_resolved_worker_bound():
    source = SOURCE.read_text(encoding="utf-8")

    assert "def _resolve_process_pool_workers(" in source
    assert "available_cpus = os.cpu_count() or 1" in source
    assert "max_workers=worker_count" in source


def test_unit_test_verifies_process_pool_bounds():
    source = TESTS.read_text(encoding="utf-8")

    assert "test_process_pool_is_capped_by_cpu_count" in source
    assert (
        "RecordingExecutor.recorded_max_workers == 4"
        in source
    )
