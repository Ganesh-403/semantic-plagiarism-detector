"""
Advanced Analytics and Performance Monitoring Module

This module provides enhanced text preprocessing, performance monitoring,
batch processing optimization, and comparison history tracking for the
Semantic Plagiarism Detector.
"""

import asyncio
import json
import re
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Set, Tuple

import numpy as np
import streamlit as st

from src.core.logging_setup import setup_logging

# Setup logging
setup_logging()
import logging

logger = logging.getLogger(__name__)

# ==============================================================================
# DATA CLASSES
# ==============================================================================


@dataclass
class ComparisonRecord:
    """Record of a document comparison operation."""

    id: str
    timestamp: str
    document_a: str
    document_b: str
    similarity_score: float
    threshold_used: float
    was_flagged: bool
    chunks_compared: int
    processing_time_ms: float
    user_id: str
    metadata: Dict[str, Any]


# ==============================================================================
# ADVANCED TEXT PREPROCESSOR
# ==============================================================================


class AdvancedTextPreprocessor:
    """Advanced text preprocessing for improved semantic comparison accuracy."""

    def __init__(self, language: str = "en"):
        self.language = language
        self.stopwords = self._load_stopwords(language)

    def _load_stopwords(self, language: str) -> Set[str]:
        """Load language-specific stopwords."""
        stopwords_sets = {
            "en": {
                "the",
                "be",
                "to",
                "of",
                "and",
                "a",
                "in",
                "that",
                "have",
                "i",
                "it",
                "for",
                "not",
                "on",
                "with",
                "he",
                "as",
                "you",
                "do",
                "at",
                "this",
                "but",
                "his",
                "by",
                "from",
                "they",
                "we",
                "say",
                "her",
                "she",
                "or",
                "an",
                "will",
                "my",
                "one",
                "all",
                "would",
                "there",
                "their",
                "what",
                "so",
                "up",
                "out",
                "if",
                "about",
                "who",
                "get",
                "which",
                "go",
                "me",
                "when",
                "make",
                "can",
                "like",
                "time",
                "no",
                "just",
                "him",
                "know",
                "take",
                "people",
                "into",
                "year",
                "your",
                "good",
                "some",
                "could",
                "them",
                "see",
                "other",
                "than",
                "then",
                "now",
                "look",
                "only",
                "come",
                "its",
                "over",
                "think",
                "also",
                "back",
                "after",
                "use",
                "two",
                "how",
                "our",
                "work",
                "first",
                "well",
                "way",
                "even",
                "new",
                "want",
                "because",
                "any",
                "these",
                "give",
                "day",
                "most",
                "us",
            },
            "es": {
                "el",
                "la",
                "los",
                "las",
                "un",
                "una",
                "unos",
                "unas",
                "de",
                "del",
                "en",
                "por",
                "para",
                "con",
                "sin",
                "sobre",
                "tras",
                "durante",
                "mediante",
                "entre",
                "hasta",
                "desde",
                "hacia",
                "a",
                "ante",
                "bajo",
                "cabe",
                "contra",
                "según",
            },
            "fr": {
                "le",
                "la",
                "les",
                "un",
                "une",
                "des",
                "du",
                "de",
                "à",
                "en",
                "pour",
                "par",
                "avec",
                "sans",
                "sur",
                "sous",
                "entre",
                "dans",
                "contre",
                "après",
                "avant",
                "pendant",
                "depuis",
                "jusque",
                "chez",
            },
        }
        return stopwords_sets.get(language, set())

    def preprocess_for_comparison(self, text: str) -> Dict[str, str]:
        """Apply comprehensive preprocessing pipeline."""
        result = {}

        # 1. Original text
        result["original"] = text

        # 2. Basic normalization
        normalized = self._normalize_text(text)
        result["normalized"] = normalized

        # 3. Remove stopwords
        no_stopwords = self._remove_stopwords(normalized)
        result["no_stopwords"] = no_stopwords

        # 4. Simple lemmatization (using word frequency-based approach)
        lemmatized = self._simple_lemmatize(no_stopwords)
        result["lemmatized"] = lemmatized

        # 5. Clean for embedding
        cleaned = self._clean_for_embedding(text)
        result["cleaned_for_embedding"] = cleaned

        return result

    def _normalize_text(self, text: str) -> str:
        """Normalize text by removing extra spaces and normalizing unicode."""
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)
        # Normalize unicode characters
        import unicodedata

        text = unicodedata.normalize("NFKD", text)
        return text.strip()

    def _remove_stopwords(self, text: str) -> str:
        """Remove stopwords from text."""
        words = text.lower().split()
        filtered = [w for w in words if w not in self.stopwords]
        return " ".join(filtered)

    def _simple_lemmatize(self, text: str) -> str:
        """Simple lemmatization using common suffixes."""
        words = text.split()
        lemmatized = []

        for word in words:
            # Common English suffixes
            if word.endswith("ing") and len(word) > 5:
                word = word[:-3]
            elif word.endswith("ed") and len(word) > 4:
                word = word[:-2]
            elif word.endswith("tion") or word.endswith("sion"):
                word = word[:-4]
            elif word.endswith("ness") or word.endswith("ment"):
                word = word[:-4]
            elif word.endswith("able") or word.endswith("ible"):
                word = word[:-4]
            elif word.endswith("ly"):
                word = word[:-2]
            elif word.endswith("ful"):
                word = word[:-3]
            elif word.endswith("ous"):
                word = word[:-3]
            lemmatized.append(word)

        return " ".join(lemmatized)

    def _clean_for_embedding(self, text: str) -> str:
        """Clean text specifically for embedding generation."""
        # Remove special characters but keep meaningful punctuation
        text = re.sub(r"[^\w\s.!?]", " ", text)
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)
        # Remove excessive punctuation
        text = re.sub(r"([.!?])\1+", r"\1", text)
        return text.strip()

    def extract_key_phrases(self, text: str, max_phrases: int = 10) -> List[str]:
        """Extract key phrases using simple frequency-based approach."""
        if not text.strip():
            return []

        words = re.findall(r"\b\w+\b", text.lower())
        word_freq = Counter(words)

        # Remove stopwords
        for stopword in self.stopwords:
            word_freq.pop(stopword, None)

        # Get top words as phrases
        top_words = [word for word, _ in word_freq.most_common(max_phrases)]
        return top_words

    def compute_readability_score(self, text: str) -> Dict[str, float]:
        """Compute readability metrics."""
        if not text.strip():
            return {
                "flesch_reading_ease": 0.0,
                "word_count": 0,
                "sentence_count": 0,
                "avg_word_length": 0,
                "avg_sentence_length": 0,
            }

        sentences = re.split(r"[.!?]+", text)
        sentences = [s for s in sentences if s.strip()]
        words = text.split()

        word_count = len(words)
        sentence_count = len(sentences)
        avg_word_length = sum(len(w) for w in words) / max(word_count, 1)
        avg_sentence_length = word_count / max(sentence_count, 1)

        # Simplified Flesch Reading Ease
        flesch = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_word_length / 10)

        return {
            "flesch_reading_ease": max(0, min(100, flesch)),
            "word_count": word_count,
            "sentence_count": sentence_count,
            "avg_word_length": avg_word_length,
            "avg_sentence_length": avg_sentence_length,
        }


# ==============================================================================
# CONTEXT-PRESERVING CHUNKER
# ==============================================================================


class ContextPreservingChunker:
    """Advanced chunking that preserves semantic context across boundaries."""

    def __init__(self, chunk_size: int = 500, overlap_size: int = 50):
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size

    def chunk_with_context(
        self, text: str, min_chunk_size: int = 100
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Chunk text while preserving sentence boundaries and context."""
        # Split into sentences
        sentences = self._split_sentences(text)

        chunks = []
        current_chunk = []
        current_size = 0

        for i, sentence in enumerate(sentences):
            sentence_size = len(sentence)

            # If a single sentence is too long, split it
            if sentence_size > self.chunk_size:
                if current_chunk:
                    # Save current chunk
                    chunk_text = " ".join(current_chunk)
                    chunks.append(
                        (
                            chunk_text,
                            self._create_chunk_metadata(
                                current_chunk, i - len(current_chunk), i
                            ),
                        )
                    )
                    current_chunk = []
                    current_size = 0

                # Split long sentence into smaller parts
                sub_chunks = self._split_long_sentence(sentence)
                for sub_chunk in sub_chunks:
                    chunks.append(
                        (sub_chunk, self._create_chunk_metadata([sub_chunk], i, i + 1))
                    )
                continue

            # Check if adding this sentence exceeds chunk size
            if current_size + sentence_size > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = " ".join(current_chunk)
                chunks.append(
                    (
                        chunk_text,
                        self._create_chunk_metadata(
                            current_chunk, i - len(current_chunk), i
                        ),
                    )
                )

                # Start new chunk with overlap
                if len(current_chunk) > 1:
                    # Include last sentence as overlap
                    overlap_sentences = current_chunk[-1:]
                    current_chunk = overlap_sentences
                    current_size = sum(len(s) for s in overlap_sentences)
                else:
                    current_chunk = []
                    current_size = 0

            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_size += sentence_size

        # Add the last chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(
                (
                    chunk_text,
                    self._create_chunk_metadata(
                        current_chunk,
                        len(sentences) - len(current_chunk),
                        len(sentences),
                    ),
                )
            )

        # Filter chunks below minimum size
        chunks = [
            (text, meta)
            for text, meta in chunks
            if len(text.split()) >= min_chunk_size // 10
        ]

        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _split_long_sentence(self, sentence: str) -> List[str]:
        """Split a long sentence into smaller chunks."""
        # Split by clauses (comma, semicolon, etc.)
        parts = re.split(r"(?<=[;,]\s)", sentence)
        if len(parts) <= 1:
            # If no clauses, split by words
            words = sentence.split()
            chunks = []
            current_chunk = []
            current_size = 0

            for word in words:
                if current_size + len(word) > self.chunk_size and current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_size = 0
                current_chunk.append(word)
                current_size += len(word) + 1

            if current_chunk:
                chunks.append(" ".join(current_chunk))
            return chunks

        # Combine parts into chunks
        chunks = []
        current_chunk = []
        current_size = 0

        for part in parts:
            part_size = len(part)
            if current_size + part_size > self.chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_size = 0
            current_chunk.append(part)
            current_size += part_size

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _create_chunk_metadata(
        self, sentences: List[str], start_idx: int, end_idx: int
    ) -> Dict[str, Any]:
        """Create metadata for a chunk."""
        return {
            "sentence_count": len(sentences),
            "start_sentence_idx": start_idx,
            "end_sentence_idx": end_idx,
            "word_count": sum(len(s.split()) for s in sentences),
            "char_count": sum(len(s) for s in sentences),
        }


# ==============================================================================
# OPTIMIZED BATCH PROCESSOR
# ==============================================================================


class ProcessingStatus:
    """Processing status constants."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class OptimizedBatchProcessor:
    """Optimized batch processing with parallel execution support."""

    def __init__(self, max_workers: int = 4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.batch_size = 10
        self.processing_status = {}

    async def process_documents_async(
        self, documents: Dict[str, bytes], processing_fn, **kwargs
    ) -> Dict[str, Any]:
        """Process documents asynchronously in parallel batches."""
        results = {}
        tasks = []

        # Create batches
        batch_generator = self._create_batches(documents.items(), self.batch_size)

        for batch in batch_generator:
            batch_dict = dict(batch)
            loop = asyncio.get_event_loop()
            task = loop.run_in_executor(
                self.executor, processing_fn, batch_dict, **kwargs
            )
            tasks.append((list(batch_dict.keys()), task))

        # Process tasks
        for doc_names, task in tasks:
            try:
                batch_result = await task
                for doc_name in doc_names:
                    results[doc_name] = batch_result.get(doc_name)
                    self.processing_status[doc_name] = ProcessingStatus.COMPLETED
            except Exception as e:
                for doc_name in doc_names:
                    self.processing_status[doc_name] = ProcessingStatus.FAILED
                logger.error(f"Batch processing failed: {e}")

        return results

    def _create_batches(self, items, batch_size: int) -> Generator:
        """Create batches from items."""
        batch = []
        for item in items:
            batch.append(item)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def get_processing_status(self) -> Dict[str, str]:
        """Get processing status for all documents."""
        return {doc_name: status for doc_name, status in self.processing_status.items()}


# ==============================================================================
# COMPARISON HISTORY MANAGER
# ==============================================================================


class ComparisonHistoryManager:
    """Manage and query comparison history."""

    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.current_file = self.storage_path / "comparison_history.jsonl"

    def save_comparison(self, record: ComparisonRecord) -> None:
        """Save a comparison record to history."""
        try:
            with open(self.current_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(record)) + "\n")
        except Exception as e:
            logger.error(f"Failed to save comparison record: {e}")

    def get_comparisons(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        document_name: Optional[str] = None,
        min_similarity: Optional[float] = None,
        max_similarity: Optional[float] = None,
        was_flagged: Optional[bool] = None,
        limit: int = 100,
    ) -> List[ComparisonRecord]:
        """Query comparison history with filters."""
        records = []
        try:
            with open(self.current_file, "r", encoding="utf-8") as f:
                for line in f:
                    if len(records) >= limit:
                        break
                    try:
                        data = json.loads(line.strip())
                        record = ComparisonRecord(**data)

                        # Apply filters
                        if start_date and record.timestamp < start_date:
                            continue
                        if end_date and record.timestamp > end_date:
                            continue
                        if document_name:
                            if document_name not in [
                                record.document_a,
                                record.document_b,
                            ]:
                                continue
                        if min_similarity and record.similarity_score < min_similarity:
                            continue
                        if max_similarity and record.similarity_score > max_similarity:
                            continue
                        if (
                            was_flagged is not None
                            and record.was_flagged != was_flagged
                        ):
                            continue

                        records.append(record)
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.error(f"Error reading comparison history: {e}")

        return records

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about comparison history."""
        records = self.get_comparisons(limit=10000)

        if not records:
            return {
                "total_comparisons": 0,
                "avg_similarity": 0.0,
                "max_similarity": 0.0,
                "min_similarity": 0.0,
                "flagged_count": 0,
                "unique_documents": set(),
            }

        similarities = [r.similarity_score for r in records]
        return {
            "total_comparisons": len(records),
            "avg_similarity": sum(similarities) / len(similarities),
            "max_similarity": max(similarities),
            "min_similarity": min(similarities),
            "flagged_count": sum(1 for r in records if r.was_flagged),
            "unique_documents": {r.document_a for r in records}.union(
                {r.document_b for r in records}
            ),
        }


# ==============================================================================
# PERFORMANCE MONITOR
# ==============================================================================


class PerformanceMonitor:
    """Monitor and track performance metrics."""

    def __init__(self):
        self.metrics = {
            "document_processing": [],
            "embedding_generation": [],
            "similarity_computation": [],
            "chunking": [],
            "ocr_extraction": [],
        }
        self.current_timings = {}

    def start_timer(self, name: str) -> None:
        """Start a timer for a specific operation."""
        self.current_timings[name] = time.perf_counter()

    def stop_timer(self, name: str) -> float:
        """Stop timer and record metrics."""
        if name not in self.current_timings:
            return 0.0

        elapsed = time.perf_counter() - self.current_timings[name]
        if name in self.metrics:
            self.metrics[name].append(elapsed)

        # Keep only last 1000 measurements
        if len(self.metrics[name]) > 1000:
            self.metrics[name] = self.metrics[name][-1000:]

        return elapsed

    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all monitored operations."""
        stats = {}
        for name, measurements in self.metrics.items():
            if measurements:
                sorted_measurements = sorted(measurements)  # noqa: F841
                stats[name] = {
                    "avg": sum(measurements) / len(measurements),
                    "min": min(measurements),
                    "max": max(measurements),
                    "p95": np.percentile(measurements, 95),
                    "p99": np.percentile(measurements, 99),
                    "count": len(measurements),
                }
        return stats

    def reset(self) -> None:
        """Reset all metrics."""
        for key in self.metrics:
            self.metrics[key] = []
        self.current_timings = {}


# ==============================================================================
# UI COMPONENTS
# ==============================================================================


def render_processing_status_widget(batch_processor: OptimizedBatchProcessor):
    """Render a widget showing current processing status."""
    status = batch_processor.get_processing_status()

    if not status:
        st.info("No active processing jobs")
        return

    completed = sum(1 for s in status.values() if s == ProcessingStatus.COMPLETED)
    failed = sum(1 for s in status.values() if s == ProcessingStatus.FAILED)
    total = len(status)
    pending = total - completed - failed

    st.markdown("### ⚡ Processing Status")

    col1, col2, col3 = st.columns(3)
    col1.metric("Completed", f"{completed}/{total}")
    col2.metric("Pending", pending)
    col3.metric("Failed", failed, delta=f"-{failed}" if failed > 0 else None)

    # Progress bar
    progress = completed / total if total > 0 else 0
    color = "green" if progress == 1 else "orange" if progress > 0.5 else "red"
    st.markdown(
        f"""
        <div style="
            background-color: #f0f0f0;
            border-radius: 10px;
            height: 20px;
            overflow: hidden;
        ">
            <div style="
                width: {progress * 100}%;
                background-color: {color};
                height: 100%;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-size: 12px;
                font-weight: bold;
                transition: width 0.5s ease;
            ">
                {progress * 100:.1f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_document_analysis_widget(
    doc_name: str, preprocessing_results: Dict[str, str]
):
    """Render a detailed document analysis widget."""
    with st.expander(f"📄 Analysis: {doc_name}", expanded=False):
        # Readability metrics
        if "readability" in preprocessing_results:
            readability = preprocessing_results["readability"]
            cols = st.columns(4)
            metrics = [
                ("Flesch Reading Ease", readability.get("flesch_reading_ease", 0)),
                ("Word Count", readability.get("word_count", 0)),
                ("Sentence Count", readability.get("sentence_count", 0)),
                ("Avg Word Length", readability.get("avg_word_length", 0)),
            ]
            for col, (label, value) in zip(cols, metrics):
                if isinstance(value, float):
                    col.metric(label, f"{value:.2f}")
                else:
                    col.metric(label, value)

        # Key phrases
        if "key_phrases" in preprocessing_results:
            phrases = preprocessing_results["key_phrases"]
            st.markdown("**Key Phrases:**")
            st.markdown(", ".join(phrases[:10]))

        # Preprocessed text preview
        if "cleaned_for_embedding" in preprocessing_results:
            preview = preprocessing_results["cleaned_for_embedding"][:500]
            st.markdown("**Cleaned Text Preview:**")
            st.caption(preview + "..." if len(preview) >= 500 else preview)


def render_performance_metrics(monitor: PerformanceMonitor):
    """Render performance metrics dashboard."""
    stats = monitor.get_statistics()

    if not stats:
        st.info("No performance metrics available yet")
        return

    st.markdown("### 📊 Performance Metrics")

    for op_name, op_stats in stats.items():
        with st.expander(f"**{op_name.replace('_', ' ').title()}**", expanded=False):
            cols = st.columns(4)
            cols[0].metric("Avg", f"{op_stats['avg'] * 1000:.1f}ms")
            cols[1].metric("P95", f"{op_stats['p95'] * 1000:.1f}ms")
            cols[2].metric("P99", f"{op_stats['p99'] * 1000:.1f}ms")
            cols[3].metric("Count", op_stats["count"])


def render_advanced_features_sidebar():
    """Render advanced features in the sidebar."""
    st.markdown("---")
    st.markdown("### 🚀 Advanced Features")

    # Performance monitoring toggle
    show_performance = st.checkbox(
        "Show Performance Metrics",
        value=False,
        key="show_performance_metrics_sidebar",
        help="Display real-time performance statistics.",
    )

    if show_performance and "performance_monitor" in st.session_state:
        monitor = st.session_state.performance_monitor
        render_performance_metrics(monitor)

    # Advanced preprocessing options
    with st.expander("🔧 Preprocessing Options", expanded=False):
        use_lemmatization = st.checkbox(
            "Use Lemmatization",
            value=True,
            key="use_lemmatization",
            help="Apply lemmatization to reduce words to their base form.",
        )

        remove_stopwords = st.checkbox(
            "Remove Stopwords",
            value=True,
            key="remove_stopwords",
            help="Remove common stopwords for better semantic comparison.",
        )

        if use_lemmatization or remove_stopwords:
            st.caption("✅ Advanced preprocessing enabled")


# ==============================================================================
# INITIALIZATION AND INTEGRATION
# ==============================================================================


def initialize_advanced_features():
    """Initialize all advanced features."""
    if "performance_monitor" not in st.session_state:
        st.session_state.performance_monitor = PerformanceMonitor()
        logger.info("Performance Monitor initialized")

    if "text_preprocessor" not in st.session_state:
        st.session_state.text_preprocessor = AdvancedTextPreprocessor()
        logger.info("Text Preprocessor initialized")

    if "batch_processor" not in st.session_state:
        st.session_state.batch_processor = OptimizedBatchProcessor()
        logger.info("Batch Processor initialized")

    if "comparison_history" not in st.session_state:
        # Determine storage path
        data_dir = Path(st.session_state.get("data_dir", "."))
        history_path = data_dir / "comparison_history"
        st.session_state.comparison_history = ComparisonHistoryManager(history_path)
        logger.info("Comparison History Manager initialized")

    if "context_chunker" not in st.session_state:
        st.session_state.context_chunker = ContextPreservingChunker()
        logger.info("Context Preserving Chunker initialized")


def track_comparison(
    document_a: str,
    document_b: str,
    similarity: float,
    threshold: float,
    processing_time_ms: float = 0.0,
):
    """Track document comparison in history."""
    if "comparison_history" not in st.session_state:
        return

    history = st.session_state.comparison_history
    record = ComparisonRecord(
        id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc).isoformat(),
        document_a=document_a,
        document_b=document_b,
        similarity_score=similarity,
        threshold_used=threshold,
        was_flagged=similarity >= threshold,
        chunks_compared=1,
        processing_time_ms=processing_time_ms,
        user_id=st.session_state.get("username", "anonymous"),
        metadata={},
    )
    history.save_comparison(record)


# ==============================================================================
# ENHANCED PIPELINE WRAPPER
# ==============================================================================


def run_pipeline_with_tracking(
    file_bytes_dict, ocr_language, ocr_dpi, chunk_size, chunk_overlap
):
    """
    Enhanced pipeline wrapper with performance tracking.

    This wraps the original run_pipeline function with performance monitoring
    and tracking capabilities.
    """
    monitor = st.session_state.get("performance_monitor")

    if monitor:
        monitor.start_timer("document_processing")

    try:
        # Import the original run_pipeline function
        from app.main import run_pipeline

        # Run the original pipeline
        result = run_pipeline(
            file_bytes_dict, ocr_language, ocr_dpi, chunk_size, chunk_overlap
        )

        if monitor:
            processing_time = monitor.stop_timer("document_processing")

            # Track comparisons if we have similarity data
            if result and len(result) >= 4:
                sim_df = result[3]  # sim_df is at index 3
                if sim_df is not None and not sim_df.empty:
                    for _, row in sim_df.iterrows():
                        track_comparison(
                            document_a=row.get("doc_a", ""),
                            document_b=row.get("doc_b", ""),
                            similarity=row.get("similarity", 0.0),
                            threshold=st.session_state.get("threshold_slider", 0.5),
                            processing_time_ms=processing_time * 1000,
                        )

        return result

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        if monitor:
            monitor.stop_timer("document_processing")
        raise
