# MIT License
#
# Copyright (c) 2026 Ganesh Kambli
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
app/components/scan_progress_tracker.py

Reusable real-time progress tracker for the document scanning / plagiarism
detection pipeline.

--------------------------------------------------------------------------
INTEGRATION NOTE (read this before wiring it in)
--------------------------------------------------------------------------
This module was written WITHOUT access to the actual repository source
(app/streamlit_app.py, src/core/processing.py, src/core/worker.py,
src/core/embedding_model.py, src/core/faiss_index.py / faiss_indexer.py,
src/core/similarity.py, app/theme.py). I could not fetch that repo, so I
could not confirm:

  - the exact function signature that iterates over uploaded files
  - whether worker.py already exposes any progress/callback mechanism
  - what app/theme.py exports (color constants, a CSS-injection helper,
    card-rendering helpers used by bulk_export.py / incident_export.py /
    faiss_results.py / time_breakdown.py)

So instead of guessing at those symbols and importing things that may not
exist (which would silently break at import time), this module:

  1. Is fully self-contained — it has a minimal built-in style palette
     that LOOKS like a plausible shared theme, clearly isolated in
     `_FALLBACK_THEME` at the top, so it's trivial to delete once you
     swap in the real `app/theme.py` values.
  2. Exposes a plain callback interface (`ProgressTracker`) that your
     existing pipeline code drives — it does NOT reach into
     processing.py/worker.py itself, so it can't duplicate or diverge
     from your real pipeline logic. You call `tracker.update_stage(...)`
     etc. from wherever your pipeline already loops over files/batches.

TO WIRE THIS UP FOR REAL:
  - Replace `from app.components.scan_progress_tracker import ProgressTracker`
    usage in streamlit_app.py's upload/scan handler.
  - Wherever processing.py (or worker.py) currently loops over files and
    calls embedding_model / faiss_index / similarity functions, add calls
    to the tracker methods at each stage transition (see the
    `# --- INTEGRATION EXAMPLE ---` block at the bottom of this file for
    the exact shape expected).
  - If app/theme.py exposes real color constants or a CSS helper, replace
    the `_FALLBACK_THEME` dict below with imports from it, and delete the
    injected `<style>` block in `_inject_css()` in favor of your shared
    one, so this component matches bulk_export.py / faiss_results.py etc.
    exactly instead of approximating it.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import streamlit as st

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Fallback theme (delete this once wired to the real app/theme.py)
# --------------------------------------------------------------------------
_FALLBACK_THEME = {
    "primary": "#4F46E5",
    "success": "#16A34A",
    "success_bg": "#F0FDF4",
    "success_border": "#BBF7D0",
    "error": "#DC2626",
    "error_bg": "#FEF2F2",
    "error_border": "#FECACA",
    "card_bg": "#FFFFFF",
    "card_border": "#E5E7EB",
    "text_muted": "#6B7280",
    "radius": "10px",
}


def _inject_css() -> None:
    """Inject scoped CSS once per session.

    Uses a session-state guard so the <style> block isn't re-injected on
    every rerun (Streamlit reruns the whole script on each interaction),
    which keeps this at near-zero overhead.
    """
    if st.session_state.get("_scan_tracker_css_injected"):
        return

    t = _FALLBACK_THEME
    st.markdown(
        f"""
        <style>
        .scan-card {{
            background: {t['card_bg']};
            border: 1px solid {t['card_border']};
            border-radius: {t['radius']};
            padding: 1rem 1.25rem;
            margin-bottom: 0.75rem;
        }}
        .scan-card-title {{
            font-weight: 600;
            font-size: 0.95rem;
            margin-bottom: 0.35rem;
        }}
        .scan-card-sub {{
            color: {t['text_muted']};
            font-size: 0.85rem;
        }}
        .scan-success-card {{
            background: {t['success_bg']};
            border: 1px solid {t['success_border']};
            border-radius: {t['radius']};
            padding: 1.25rem;
            text-align: center;
        }}
        .scan-success-title {{
            color: {t['success']};
            font-weight: 700;
            font-size: 1.15rem;
            margin-bottom: 0.5rem;
        }}
        .scan-error-card {{
            background: {t['error_bg']};
            border: 1px solid {t['error_border']};
            border-radius: {t['radius']};
            padding: 1.1rem;
        }}
        .scan-error-title {{
            color: {t['error']};
            font-weight: 700;
            margin-bottom: 0.35rem;
        }}
        .scan-stat-row {{
            display: flex;
            gap: 2rem;
            margin-top: 0.5rem;
        }}
        .scan-stat-value {{
            font-size: 1.4rem;
            font-weight: 700;
        }}
        .scan-stat-label {{
            color: {t['text_muted']};
            font-size: 0.8rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.session_state["_scan_tracker_css_injected"] = True


# --------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------
class ScanStage(str, Enum):
    READING = "Reading documents"
    EXTRACTING = "Extracting text"
    CHUNKING = "Chunking text"
    EMBEDDING = "Generating embeddings"
    INDEXING = "Building FAISS index"
    SIMILARITY = "Similarity analysis"
    SAVING = "Saving results"
    COMPLETED = "Completed"


@dataclass
class _TrackerState:
    total_files: int = 0
    current_file_index: int = 0
    current_file_name: str = ""
    stage: ScanStage = ScanStage.READING
    start_time: float = field(default_factory=time.time)

    embedding_current_batch: int = 0
    embedding_total_batches: int = 0

    faiss_indexed_vectors: int = 0
    faiss_total_vectors: int = 0

    documents_processed: int = 0
    incidents_detected: int = 0

    finished: bool = False
    errored: bool = False
    error_message: str = ""
    error_stage: Optional[ScanStage] = None


def _format_duration(seconds: float) -> str:
    """Format seconds as e.g. '2m 31s' or '48s'."""
    seconds = max(0, int(seconds))
    minutes, secs = divmod(seconds, 60)
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


class ProgressTracker:
    """Real-time progress tracker for the document scanning pipeline.

    Usage pattern (called from wherever the actual pipeline loop lives,
    e.g. processing.py's file loop / worker.py's job runner):

        tracker = ProgressTracker(total_files=len(uploaded_files))
        tracker.start()
        try:
            for i, file in enumerate(uploaded_files):
                tracker.update_file(i + 1, file.name)

                tracker.update_stage(ScanStage.READING)
                ... existing read logic ...

                tracker.update_stage(ScanStage.EXTRACTING)
                ... existing extract logic ...

                tracker.update_stage(ScanStage.EMBEDDING)
                for batch_i, batch in enumerate(batches):
                    ... existing embedding logic ...
                    tracker.update_embedding_progress(batch_i + 1, len(batches))

            tracker.update_stage(ScanStage.INDEXING)
            ... existing FAISS build logic, calling
                tracker.update_faiss_progress(indexed, total) periodically ...

            tracker.update_stage(ScanStage.SIMILARITY)
            ... existing similarity logic ...

            tracker.update_stage(ScanStage.SAVING)
            ... existing save logic ...

            tracker.finish(documents=len(uploaded_files), incidents=incident_count)
        except Exception as exc:
            tracker.error(str(exc))
            raise

    All render calls target a single st.empty() placeholder, so the UI is
    replaced in place rather than appended to — the progress bar can never
    get "stuck" duplicated on screen, and reruns only redraw when a value
    actually changed (see `_dirty` guard in `_render`).
    """

    def __init__(
        self,
        total_files: int,
        container: Optional["st.delta_generator.DeltaGenerator"] = None,
    ) -> None:
        _inject_css()
        self._state = _TrackerState(total_files=total_files)
        self._placeholder = container.empty() if container is not None else st.empty()
        self._last_rendered_signature: Optional[tuple] = None

    # ---------------------------------------------------------------- #
    # Public API
    # ---------------------------------------------------------------- #
    def start(self) -> None:
        """Begin tracking. Resets the timer and renders the initial state."""
        self._state = _TrackerState(total_files=self._state.total_files)
        logger.info("Scan started: %d file(s)", self._state.total_files)
        self._render()

    def update_file(self, file_index: int, file_name: str) -> None:
        """Advance to a new file. `file_index` is 1-based."""
        self._state.current_file_index = file_index
        self._state.current_file_name = file_name
        logger.debug(
            "Now processing file %d/%d: %s",
            file_index,
            self._state.total_files,
            file_name,
        )
        self._render()

    def update_stage(self, stage: ScanStage) -> None:
        """Move to a new pipeline stage."""
        self._state.stage = stage
        logger.debug("Stage -> %s", stage.value)
        self._render()

    def update_embedding_progress(self, current_batch: int, total_batches: int) -> None:
        """Report embedding-generation batch progress."""
        self._state.embedding_current_batch = current_batch
        self._state.embedding_total_batches = total_batches
        self._render()

    def update_faiss_progress(self, indexed_vectors: int, total_vectors: int) -> None:
        """Report FAISS index build progress."""
        self._state.faiss_indexed_vectors = indexed_vectors
        self._state.faiss_total_vectors = total_vectors
        self._render()

    def update_eta(self) -> str:
        """Compute and return the current ETA string (also used internally)."""
        return self._compute_eta_string()

    def update(self, **kwargs) -> None:
        """Generic bulk-update escape hatch for any tracked field.

        Lets calling code update several fields in one pass without
        triggering multiple intermediate renders, e.g.:
            tracker.update(current_file_name="paper3.pdf", stage=ScanStage.CHUNKING)
        """
        for key, value in kwargs.items():
            if hasattr(self._state, key):
                setattr(self._state, key, value)
            else:
                logger.warning(
                    "ProgressTracker.update: unknown field '%s' ignored", key
                )
        self._render()

    def finish(self, documents: int, incidents: int) -> None:
        """Mark the scan complete and render the success card."""
        self._state.finished = True
        self._state.stage = ScanStage.COMPLETED
        self._state.documents_processed = documents
        self._state.incidents_detected = incidents
        logger.info("Scan complete: %d documents, %d incidents", documents, incidents)
        self._render(force=True)

    def error(self, message: str, retry_callback: Optional[callable] = None) -> None:
        """Mark the scan as failed and render the error card with a retry option.

        `retry_callback`, if given, is invoked when the user clicks Retry
        (it should re-kick the pipeline — e.g. re-call the function that
        drives this tracker).
        """
        self._state.errored = True
        self._state.error_message = message
        self._state.error_stage = self._state.stage
        logger.error("Scan failed at stage '%s': %s", self._state.stage.value, message)
        self._render(force=True)

        if retry_callback is not None:
            retry_key = f"_scan_tracker_retry_{id(self)}"
            if self._placeholder.button("Retry", key=retry_key):
                retry_callback()

    # ---------------------------------------------------------------- #
    # Internal rendering
    # ---------------------------------------------------------------- #
    def _compute_eta_string(self) -> str:
        elapsed = time.time() - self._state.start_time
        completed = max(self._state.current_file_index - 1, 0)
        total = max(self._state.total_files, 1)

        if completed == 0 or self._state.finished:
            return "Calculating..." if not self._state.finished else "0s"

        avg_per_file = elapsed / completed
        remaining_files = max(total - completed, 0)
        eta_seconds = avg_per_file * remaining_files
        return _format_duration(eta_seconds)

    def _overall_fraction(self) -> float:
        total = max(self._state.total_files, 1)
        # Treat the file currently in progress as partially done so the
        # bar isn't stuck at the previous file's completion percentage.
        fraction = (
            (self._state.current_file_index - 0.5) / total
            if self._state.current_file_index
            else 0.0
        )
        return min(max(fraction, 0.0), 1.0)

    def _signature(self) -> tuple:
        """Cheap tuple of everything that affects rendering — used to skip
        redundant re-renders when nothing has actually changed."""
        s = self._state
        return (
            s.total_files,
            s.current_file_index,
            s.current_file_name,
            s.stage,
            s.embedding_current_batch,
            s.embedding_total_batches,
            s.faiss_indexed_vectors,
            s.faiss_total_vectors,
            s.finished,
            s.errored,
            s.error_message,
            int(time.time() - s.start_time),  # ~1s resolution, keeps ETA/elapsed fresh
        )

    def _render(self, force: bool = False) -> None:
        signature = self._signature()
        if not force and signature == self._last_rendered_signature:
            return
        self._last_rendered_signature = signature

        with self._placeholder.container():
            if self._state.errored:
                self._render_error()
            elif self._state.finished:
                self._render_success()
            else:
                self._render_in_progress()

    def _render_in_progress(self) -> None:
        s = self._state
        elapsed = time.time() - s.start_time
        overall_pct = self._overall_fraction()

        st.markdown(f"**Processing {s.current_file_index} of {s.total_files} files**")
        st.progress(overall_pct, text=f"{int(overall_pct * 100)}%")

        st.markdown(
            f"""
            <div class="scan-card">
                <div class="scan-card-title">📄 {s.current_file_name or 'Waiting for file...'}</div>
                <div class="scan-card-sub">
                    Document {s.current_file_index} of {s.total_files} &nbsp;·&nbsp;
                    Stage: {s.stage.value} &nbsp;·&nbsp;
                    Elapsed: {_format_duration(elapsed)}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if s.stage == ScanStage.EMBEDDING and s.embedding_total_batches:
            frac = s.embedding_current_batch / max(s.embedding_total_batches, 1)
            st.markdown("**Generating embeddings...**")
            st.progress(
                min(max(frac, 0.0), 1.0),
                text=f"Batch {s.embedding_current_batch} of {s.embedding_total_batches}",
            )

        if s.stage == ScanStage.INDEXING and s.faiss_total_vectors:
            frac = s.faiss_indexed_vectors / max(s.faiss_total_vectors, 1)
            st.markdown("**Building FAISS Index**")
            st.progress(
                min(max(frac, 0.0), 1.0),
                text=f"{s.faiss_indexed_vectors} of {s.faiss_total_vectors} vectors",
            )

        st.caption(f"⏱️ Estimated time remaining: {self._compute_eta_string()}")

    def _render_success(self) -> None:
        s = self._state
        st.markdown(
            f"""
            <div class="scan-success-card">
                <div class="scan-success-title">✅ Scan Complete</div>
                <div class="scan-stat-row" style="justify-content: center;">
                    <div>
                        <div class="scan-stat-value">{s.documents_processed}</div>
                        <div class="scan-stat-label">Documents Processed</div>
                    </div>
                    <div>
                        <div class="scan-stat-value">{s.incidents_detected}</div>
                        <div class="scan-stat-label">Incidents Detected</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    def _render_error(self) -> None:
        s = self._state
        stage_label = s.error_stage.value if s.error_stage else "Unknown"
        st.markdown(
            f"""
            <div class="scan-error-card">
                <div class="scan-error-title">⚠️ Scan Failed</div>
                <div class="scan-card-sub">Stage: {stage_label}</div>
                <div style="margin-top: 0.5rem;">{s.error_message}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# --------------------------------------------------------------------------
# INTEGRATION EXAMPLE (not executed — for reference when wiring into
# streamlit_app.py / processing.py / worker.py)
# --------------------------------------------------------------------------
def _integration_example() -> None:  # pragma: no cover
    """
    from app.components.scan_progress_tracker import ProgressTracker, ScanStage
    # from src.core.processing import process_uploaded_files  # your real pipeline

    uploaded_files = st.file_uploader("Upload documents", accept_multiple_files=True)

    if uploaded_files and st.button("Run Scan"):
        tracker = ProgressTracker(total_files=len(uploaded_files))
        tracker.start()

        def run_scan():
            try:
                # Replace this block with calls into your real
                # processing.py / worker.py pipeline, inserting the
                # tracker.update_*() calls at the matching points.
                for i, f in enumerate(uploaded_files):
                    tracker.update_file(i + 1, f.name)
                    tracker.update_stage(ScanStage.READING)
                    # ... existing read logic ...
                    tracker.update_stage(ScanStage.EXTRACTING)
                    # ... existing extract logic ...

                tracker.update_stage(ScanStage.EMBEDDING)
                # ... existing embedding logic, calling
                #     tracker.update_embedding_progress(i, total) per batch ...

                tracker.update_stage(ScanStage.INDEXING)
                # ... existing FAISS build logic, calling
                #     tracker.update_faiss_progress(indexed, total) periodically ...

                tracker.update_stage(ScanStage.SIMILARITY)
                # ... existing similarity logic ...

                tracker.update_stage(ScanStage.SAVING)
                # ... existing save logic ...

                tracker.finish(documents=len(uploaded_files), incidents=0)
            except Exception as exc:  # noqa: BLE001
                tracker.error(str(exc), retry_callback=run_scan)

        run_scan()
    """
