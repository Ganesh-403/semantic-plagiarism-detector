"""
tests/core/test_metrics.py
--------------------------
Tests for the Prometheus instrumentation module.

Issue #2181: ``sync_telemetry_gauges()`` used to write an absolute document
count into the ``documents_total`` **Counter** via the private
``._value.set()`` API. Counters are required to be monotonically
non-decreasing, so deleting documents made the series drop -- which ``rate()``
and ``increase()`` read as a counter reset, inventing a spike of the full
post-reset value every time.
"""

from unittest.mock import patch

import pytest

from src.core import metrics


def _sample_value(metric, **labels):
    """Read a metric's current value straight from the client registry."""
    return metric.labels(**labels)._value.get() if labels else metric._value.get()


@pytest.fixture
def fake_telemetry():
    """Patch TelemetryService so the counts are ours to control."""
    with patch("src.core.telemetry.TelemetryService") as service:
        service.get_document_count.return_value = 10
        service.get_active_user_count.return_value = 3
        yield service


# ── the regression ─────────────────────────────────────────────────────────────


def test_sync_does_not_write_to_the_documents_counter(fake_telemetry):
    """The Counter must be left entirely alone by the sync."""
    before = _sample_value(metrics.documents_total)

    metrics.sync_telemetry_gauges()

    assert _sample_value(metrics.documents_total) == before


def test_counter_never_decreases_when_the_corpus_shrinks(fake_telemetry):
    """Ingest 10, delete 6, sync -- the counter must not fall to 4."""
    metrics.record_documents(10)
    after_ingest = _sample_value(metrics.documents_total)

    fake_telemetry.get_document_count.return_value = 4
    metrics.sync_telemetry_gauges()

    assert _sample_value(metrics.documents_total) == after_ingest
    assert _sample_value(metrics.documents_total) >= after_ingest


def test_gauge_tracks_the_current_corpus_size(fake_telemetry):
    """The gauge is what is allowed to go down."""
    fake_telemetry.get_document_count.return_value = 10
    metrics.sync_telemetry_gauges()
    assert _sample_value(metrics.corpus_documents_gauge) == 10

    fake_telemetry.get_document_count.return_value = 4
    metrics.sync_telemetry_gauges()
    assert _sample_value(metrics.corpus_documents_gauge) == 4


def test_active_user_count_is_actually_recorded(fake_telemetry):
    """The call was previously made and its result discarded."""
    fake_telemetry.get_active_user_count.return_value = 7

    metrics.sync_telemetry_gauges()

    assert _sample_value(metrics.active_users_gauge) == 7


# ── file-size gauges ───────────────────────────────────────────────────────────


def test_corpus_and_index_size_gauges_are_populated(fake_telemetry, tmp_path):
    corpus_db = tmp_path / "corpus.db"
    corpus_db.write_bytes(b"x" * 2048)
    index_file = tmp_path / "corpus.index"
    index_file.write_bytes(b"y" * 4096)

    with (
        patch("src.db.incidents.DEFAULT_DB_PATH", str(corpus_db)),
        patch("src.core.app_config.FAISS_INDEX_PATH", index_file),
    ):
        metrics.sync_telemetry_gauges()

    assert _sample_value(metrics.corpus_size_gauge) == 2048
    assert _sample_value(metrics.index_size_gauge) == 4096


def test_missing_files_do_not_raise(fake_telemetry, tmp_path):
    """A fresh install has no index file yet; the sync must still complete."""
    missing_db = tmp_path / "nope.db"
    missing_index = tmp_path / "nope.index"

    with (
        patch("src.db.incidents.DEFAULT_DB_PATH", str(missing_db)),
        patch("src.core.app_config.FAISS_INDEX_PATH", missing_index),
    ):
        metrics.sync_telemetry_gauges()  # must not raise


# ── resilience ─────────────────────────────────────────────────────────────────


def test_a_failing_telemetry_call_does_not_abort_the_rest(fake_telemetry, caplog):
    """One broken lookup must not stop the other gauges from syncing."""
    fake_telemetry.get_document_count.side_effect = RuntimeError("db down")
    fake_telemetry.get_active_user_count.return_value = 5

    metrics.sync_telemetry_gauges()

    assert _sample_value(metrics.active_users_gauge) == 5


def test_failure_is_logged_with_the_metric_name(fake_telemetry, caplog):
    """A silent breakage should at least name the metric that broke."""
    fake_telemetry.get_document_count.side_effect = RuntimeError("db down")

    with caplog.at_level("WARNING", logger="src.core.metrics"):
        metrics.sync_telemetry_gauges()

    assert "corpus_documents" in caplog.text


# ── existing helpers still behave ──────────────────────────────────────────────


def test_record_documents_increments_the_counter():
    before = _sample_value(metrics.documents_total)

    metrics.record_documents(3)

    assert _sample_value(metrics.documents_total) == before + 3


def test_record_upload_uses_the_status_label():
    before = _sample_value(metrics.uploads_total, status="success")

    metrics.record_upload("success")

    assert _sample_value(metrics.uploads_total, status="success") == before + 1


def test_record_incidents_increments_the_counter():
    before = _sample_value(metrics.flagged_incidents_total)

    metrics.record_incidents(2)

    assert _sample_value(metrics.flagged_incidents_total) == before + 2


def test_timed_decorator_observes_the_stage_histogram():
    @metrics.timed("unit_test_stage")
    def work():
        return "done"

    assert work() == "done"

    samples = metrics.pipeline_duration_seconds.labels(stage="unit_test_stage")
    assert samples._sum.get() >= 0


def test_timed_decorator_records_even_when_the_stage_raises():
    """The observation happens in a finally block, so failures are timed too."""

    @metrics.timed("unit_test_failing_stage")
    def boom():
        raise ValueError("nope")

    with pytest.raises(ValueError):
        boom()

    payload = metrics.generate_metrics_json()
    counts = [
        sample["value"]
        for sample in payload["pipeline_duration_seconds"]["metrics"]
        if sample["labels"].get("stage") == "unit_test_failing_stage"
        and sample["labels"].get("le") is None
    ]

    assert 1.0 in counts


def test_timed_decorator_preserves_metadata():
    @metrics.timed("unit_test_meta")
    def documented():
        """A docstring."""

    assert documented.__name__ == "documented"
    assert documented.__doc__ == "A docstring."


# ── JSON output ────────────────────────────────────────────────────────────────


def test_generate_metrics_json_includes_the_new_gauges(fake_telemetry):
    metrics.sync_telemetry_gauges()

    payload = metrics.generate_metrics_json()

    assert "corpus_documents" in payload
    assert "active_users" in payload
    assert payload["corpus_documents"]["type"] == "gauge"


def test_generate_metrics_json_reports_the_counter_as_a_counter():
    payload = metrics.generate_metrics_json()

    assert payload["documents"]["type"] == "counter"
