"""Prometheus metrics instrumentation for the plagiarism detector pipeline.

Exposes counters, gauges, and histograms for each pipeline stage so that
operators can monitor performance, detect regressions, and capacity-plan
via Prometheus + Grafana.
"""

from __future__ import annotations

import functools
import logging
import os
import threading
import time
from typing import Any, Callable

from prometheus_client import REGISTRY, Counter, Gauge, Histogram, generate_latest
from prometheus_client import Counter, Gauge, Histogram
from prometheus_client import generate_latest as _prometheus_generate_latest

logger = logging.getLogger(__name__)

# Configurable environment variable to completely disable metrics collection.
# If set to False, no metrics will be registered with the global collector
# and the /metrics endpoints will return a 404 Not Found error.
PROMETHEUS_METRICS_ENABLED = os.environ.get('PROMETHEUS_METRICS_ENABLED', 'True').lower() in ('true', '1', 't', 'yes')

# The registry controls exposure. If enabled, it binds to the global prometheus REGISTRY.
# If disabled, it binds to None, keeping metrics isolated and hidden from HTTP scrape endpoints.
_registry = REGISTRY if PROMETHEUS_METRICS_ENABLED else None


# ── Counters ───────────────────────────────────────────────────────────────────

documents_total = Counter(
    "documents_total",
    registry=_registry,
    "spd_documents_total",
    "Cumulative number of documents ingested since process start. "
    "Monotonic: use rate()/increase() on this. For the current corpus size "
    "see the corpus_documents gauge.",
)

flagged_incidents_total = Counter(
    "flagged_incidents_total",
    registry=_registry,
    "spd_flagged_incidents_total",
    "Total number of flagged plagiarism incidents",
)

plagiarism_incidents_total = Counter(
    "spd_plagiarism_incidents_total",
    "Total plagiarism incidents flagged",
    ["severity"],
)

uploads_total = Counter(
    "uploads_total",
    registry=_registry,
    "spd_uploads_total",
    "Total number of file upload batches processed",
    labelnames=["status"],
)

cache_hits_total = Counter(
    "spd_cache_hits_total",
    "Total cache hits",
    labelnames=["cache_type"],
)

cache_misses_total = Counter(
    "spd_cache_misses_total",
    "Total cache misses",
    labelnames=["cache_type"],
)

ocr_invocations_total = Counter(
    "spd_ocr_invocations_total",
    "Total OCR extraction attempts",
    ["status"],
)

# ── Gauges ─────────────────────────────────────────────────────────────────────

corpus_size_gauge = Gauge(
    "corpus_size_bytes",
    registry=_registry,
    "spd_corpus_size_bytes",
    "Total size on disk of the corpus database",
)

index_size_gauge = Gauge(
    "index_size_bytes",
    registry=_registry,
    "spd_index_size_bytes",
    "Total size on disk of the FAISS index file",
)

corpus_documents_gauge = Gauge(
    "corpus_documents",
    registry=_registry,
    "spd_corpus_documents",
    "Current number of documents in the corpus. Goes down when documents are "
    "deleted, which is why this is a gauge and not documents_total.",
)

active_users_gauge = Gauge(
    "active_users",
    registry=_registry,
    "spd_active_users",
    "Current number of active users",
)

active_threads_gauge = Gauge(
    "spd_active_threads",
    "Active Python threads",
)

faiss_vectors_gauge = Gauge(
    "spd_faiss_vectors_total",
    "Number of vectors in FAISS index",
)
# ── Histograms ─────────────────────────────────────────────────────────────────

pipeline_duration_seconds = Histogram(
    "pipeline_duration_seconds",
    registry=_registry,
    "spd_pipeline_duration_seconds",
    "Duration of each pipeline stage in seconds",
    labelnames=["stage"],
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
)

spd_scan_duration_seconds = Histogram(
    "spd_scan_duration_seconds",
    "Scan stage duration in seconds",
    ["stage"],
)

spd_doc_parse_seconds = Histogram(
    "spd_doc_parse_seconds",
    "Document parsing time in seconds",
    ["extension"],
)

doc_parse_seconds = spd_doc_parse_seconds

query_response_time_seconds = Histogram(
    "query_response_time_seconds",
    registry=_registry,
    "spd_query_response_time_seconds",
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


def generate_latest(*args: Any, **kwargs: Any) -> bytes:
    """Prometheus text exposition; refresh thread count before each scrape."""
    active_threads_gauge.set(threading.active_count())
    return _prometheus_generate_latest(*args, **kwargs)


def generate_metrics_json() -> dict[str, Any]:
    """Return all metrics as a JSON-serialisable dict for non-Prometheus consumers."""
    if not PROMETHEUS_METRICS_ENABLED:
        return {}
    from prometheus_client.parser import text_string_to_metric_families

    raw = generate_latest().decode("utf-8")
    families = list(text_string_to_metric_families(raw))

    metrics: dict[str, Any] = {}
    for family in families:
        samples = []
        for sample in family.samples:
            samples.append(
                {
                    "name": sample.name,
                    "labels": sample.labels,
                    "value": sample.value,
                }
            )
        metrics[family.name] = {
            "name": family.name,
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
    keep :data:`corpus_documents_gauge`, :data:`active_users_gauge`,
    :data:`corpus_size_gauge` and :data:`index_size_gauge` in sync with the
    database state.

    Note:
        This deliberately does **not** touch :data:`documents_total`. That is a
        Counter, and the Prometheus data model requires counters to be
        monotonically non-decreasing. Writing an absolute document count into
        it made the series drop whenever documents were deleted, which
        ``rate()`` and ``increase()`` interpret as a counter reset -- producing
        a phantom spike of the full post-reset value on every deletion.
    """
    from src.core.telemetry import TelemetryService

    try:
        corpus_documents_gauge.set(TelemetryService.get_document_count())
    except Exception as exc:
        logger.warning("Failed to sync 'corpus_documents' gauge: %s", exc)

    try:
        active_users_gauge.set(TelemetryService.get_active_user_count())
    except Exception as exc:
        logger.warning("Failed to sync 'active_users' gauge: %s", exc)

    # Corpus DB size
    from src.db.incidents import DEFAULT_DB_PATH as corpus_db_path

    try:
        corpus_size_gauge.set(os.path.getsize(str(corpus_db_path)))
    except OSError as exc:
        logger.debug("Could not read corpus DB size for 'corpus_size_bytes': %s", exc)

    # FAISS index size -- declared since the module was introduced but never
    # populated, so index_size_bytes always reported 0.
    from src.core.app_config import FAISS_INDEX_PATH

    try:
        index_size_gauge.set(os.path.getsize(str(FAISS_INDEX_PATH)))
    except OSError as exc:
        logger.debug("Could not read FAISS index size for 'index_size_bytes': %s", exc)
