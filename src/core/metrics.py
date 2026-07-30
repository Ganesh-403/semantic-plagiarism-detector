"""Prometheus metrics instrumentation for the plagiarism detector pipeline.

Exposes counters, gauges, and histograms for each pipeline stage so that
operators can monitor performance, detect regressions, and capacity-plan
via Prometheus + Grafana.
"""

from __future__ import annotations

import functools
import logging
import os
import time
from typing import Any, Callable, Optional

from prometheus_client import Counter, Gauge, Histogram, generate_latest

logger = logging.getLogger(__name__)

# ── Counters ───────────────────────────────────────────────────────────────────

documents_total = Counter(
    "documents_total",
    "Total number of documents in the corpus",
)

flagged_incidents_total = Counter(
    "flagged_incidents_total",
    "Total number of flagged plagiarism incidents",
)

uploads_total = Counter(
    "uploads_total",
    "Total number of file upload batches processed",
    labelnames=["status"],
)

# ── Gauges ─────────────────────────────────────────────────────────────────────

corpus_size_gauge = Gauge(
    "corpus_size_bytes",
    "Total size on disk of the corpus database",
)

index_size_gauge = Gauge(
    "index_size_bytes",
    "Total size on disk of the FAISS index file",
)

# ── Histograms ─────────────────────────────────────────────────────────────────

pipeline_duration_seconds = Histogram(
    "pipeline_duration_seconds",
    "Duration of each pipeline stage in seconds",
    labelnames=["stage"],
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
)

query_response_time_seconds = Histogram(
    "query_response_time_seconds",
    "Duration of similarity search queries in seconds",
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0),
)

# ── Timed decorator ────────────────────────────────────────────────────────────


def timed(stage: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that records the duration of a pipeline stage.

    Usage::

        @timed("embed")
        def embed_documents(chunks):
            ...

    The duration histogram is automatically labelled with *stage*.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                elapsed = time.perf_counter() - start
                pipeline_duration_seconds.labels(stage=stage).observe(elapsed)

        return wrapper

    return decorator


# ── JSON-formatted output (for non-Prometheus setups) ──────────────────────────


def generate_metrics_json() -> dict[str, Any]:
    """Return all metrics as a JSON-serialisable dict for non-Prometheus consumers."""
    from prometheus_client.parser import text_string_to_metric_families

    raw = generate_latest().decode("utf-8")
    families = list(text_string_to_metric_families(raw))

    metrics: dict[str, Any] = {}
    for family in families:
        samples = []
        for sample in family.samples:
            samples.append({
                "labels": sample.labels,
                "value": sample.value,
            })
        metrics[family.name] = {
            "type": family.type,
            "help": family.documentation,
            "metrics": samples,
        }
    return metrics


# ── Convenience helpers ────────────────────────────────────────────────────────


def record_upload(status: str = "success") -> None:
    """Increment the upload counter with the given status label."""
    uploads_total.labels(status=status).inc()


def record_documents(count: int = 1) -> None:
    """Increment the document counter by *count*."""
    documents_total.inc(count)


def record_incidents(count: int = 1) -> None:
    """Increment the incident counter by *count*."""
    flagged_incidents_total.inc(count)


def sync_telemetry_gauges() -> None:
    """Pull current counts from :class:`TelemetryService` into Prometheus gauges.

    Call this periodically (e.g. via a background thread or cron trigger) to
    keep :data:`corpus_size_gauge`, :data:`documents_total`, and other gauges
    in sync with the database state.
    """
    from src.core.telemetry import TelemetryService

    try:
        doc_count = TelemetryService.get_document_count()
        # documents_total is a Counter; we set the gauge instead
        documents_total._value.set(doc_count)  # type: ignore[attr-defined]
    except Exception as exc:
        logger.warning("Failed to sync document count gauge: %s", exc)

    try:
        user_count = TelemetryService.get_active_user_count()
    except Exception as exc:
        logger.warning("Failed to sync user count: %s", exc)

    # Corpus DB size
    from src.db.incidents import DEFAULT_DB_PATH as corpus_db_path

    try:
        size = os.path.getsize(str(corpus_db_path))
        corpus_size_gauge.set(size)
    except OSError as exc:
        logger.debug("Could not read corpus DB size: %s", exc)
