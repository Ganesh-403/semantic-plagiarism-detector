"""
Intelligent Semantic Search Engine

Features:
- Natural language search queries
- Semantic understanding of search intent
- Context-aware result ranking
- Cross-document pattern discovery
- Search history and learning
- Real-time result filtering
"""

import hashlib
import json
import re
import time
from collections import Counter, defaultdict  # noqa: F401
from dataclasses import asdict, dataclass, field  # noqa: F401
from datetime import datetime, timedelta  # noqa: F401
from pathlib import Path  # noqa: F401
from typing import Any, Dict, List, Optional, Set, Tuple  # noqa: F401

import numpy as np
import pandas as pd
import plotly.graph_objects as go  # noqa: F401
import streamlit as st
from plotly.subplots import make_subplots  # noqa: F401

# ML Libraries for NLP
try:
    from sentence_transformers import SentenceTransformer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import nltk
    from nltk.corpus import stopwords  # noqa: F401
    from nltk.tokenize import sent_tokenize, word_tokenize  # noqa: F401

    NLTK_AVAILABLE = True
    # Download required NLTK data
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)
except ImportError:
    NLTK_AVAILABLE = False

try:
    import spacy  # noqa: F401

    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False


# ==============================================================================
# DATA CLASSES
# ==============================================================================


@dataclass
class SearchResult:
    """Search result entry."""

    document: str
    chunk_text: str
    similarity_score: float
    chunk_index: int
    matched_terms: List[str] = field(default_factory=list)
    context: str = ""
    snippet: str = ""
    relevance_score: float = 0.0


@dataclass
class SearchQuery:
    """Search query record."""

    id: str
    query_text: str
    query_type: str  # semantic, keyword, hybrid
    timestamp: float
    user: str
    results_count: int
    execution_time: float
    filters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchPattern:
    """Discovered search pattern."""

    pattern_type: str  # plagiarism, similarity, keyword
    keywords: List[str]
    documents_affected: List[str]
    similarity_range: Tuple[float, float]
    frequency: int
    last_detected: float


# ==============================================================================
# SMART SEARCH ENGINE
# ==============================================================================


class SmartSearchEngine:
    """
    Intelligent semantic search engine for plagiarism detection.
    """

    def __init__(self, embedding_model: Optional[Any] = None):
        self.embedding_model = embedding_model
        self.search_index: Dict[str, np.ndarray] = {}
        self.chunk_registry: Dict[str, Dict] = {}
        self.search_history: List[SearchQuery] = []
        self.result_cache: Dict[str, List[SearchResult]] = {}
        self.pattern_cache: Dict[str, SearchPattern] = {}
        self.query_suggestions: List[str] = []
        self._initialize_model()

    def _initialize_model(self):
        """Initialize embedding model for semantic search."""
        if self.embedding_model is None and TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception as e:
                print(f"Failed to load embedding model: {e}")

    def index_documents(self, documents: Dict[str, str], chunks: Dict[str, List[str]]):
        """
        Index documents for search.

        Args:
            documents: Document name -> full text
            chunks: Document name -> list of chunks
        """
        if not TRANSFORMERS_AVAILABLE or self.embedding_model is None:
            print("Transformers not available, using fallback indexing")
            self._index_fallback(documents, chunks)
            return

        self.search_index.clear()
        self.chunk_registry.clear()

        for doc_name, doc_text in documents.items():
            # Generate embedding for full document
            doc_embedding = self.embedding_model.encode([doc_text])[0]
            self.search_index[f"{doc_name}_full"] = doc_embedding

            # Generate embeddings for chunks
            doc_chunks = chunks.get(doc_name, [])
            for idx, chunk in enumerate(doc_chunks):
                chunk_embedding = self.embedding_model.encode([chunk])[0]
                key = f"{doc_name}_chunk_{idx}"
                self.search_index[key] = chunk_embedding
                self.chunk_registry[key] = {
                    "document": doc_name,
                    "chunk_index": idx,
                    "text": chunk,
                }

    def _index_fallback(self, documents: Dict[str, str], chunks: Dict[str, List[str]]):
        """Fallback indexing when transformers not available."""
        # Simple tf-idf like indexing using word frequency
        self.search_index.clear()
        self.chunk_registry.clear()

        for doc_name, doc_text in documents.items():
            # Create simple document vector
            words = re.findall(r"\w+", doc_text.lower())
            word_freq = Counter(words)

            # Store as list of top words
            vector = [word for word, freq in word_freq.most_common(50)]
            self.search_index[f"{doc_name}_full"] = np.array(vector)

            # Index chunks
            doc_chunks = chunks.get(doc_name, [])
            for idx, chunk in enumerate(doc_chunks):
                words = re.findall(r"\w+", chunk.lower())
                word_freq = Counter(words)
                vector = [word for word, freq in word_freq.most_common(20)]
                key = f"{doc_name}_chunk_{idx}"
                self.search_index[key] = np.array(vector)
                self.chunk_registry[key] = {
                    "document": doc_name,
                    "chunk_index": idx,
                    "text": chunk,
                }

    def search(
        self, query: str, top_k: int = 10, filters: Dict[str, Any] = None
    ) -> List[SearchResult]:
        """
        Perform semantic search.

        Args:
            query: Search query
            top_k: Number of results to return
            filters: Filters to apply (document_type, date, etc.)

        Returns:
            List[SearchResult]: Search results
        """
        start_time = time.time()

        # Check cache
        cache_key = hashlib.md5(
            f"{query}_{top_k}_{json.dumps(filters or {})}".encode()
        ).hexdigest()
        if cache_key in self.result_cache:
            return self.result_cache[cache_key]

        # Process query
        processed_query = self._process_query(query)

        # Generate query embedding
        query_embedding = self._get_query_embedding(processed_query)

        # Search index
        results = []
        for key, doc_embedding in self.search_index.items():
            similarity = self._calculate_similarity(query_embedding, doc_embedding)

            if similarity > 0.1:  # Minimum threshold
                registry_entry = self.chunk_registry.get(key, {})
                result = SearchResult(
                    document=registry_entry.get("document", key),
                    chunk_text=registry_entry.get("text", ""),
                    similarity_score=similarity,
                    chunk_index=registry_entry.get("chunk_index", -1),
                    matched_terms=self._extract_matched_terms(
                        processed_query, registry_entry.get("text", "")
                    ),
                    context=self._extract_context(
                        registry_entry.get("text", ""), processed_query
                    ),
                )
                results.append(result)

        # Apply filters
        if filters:
            results = self._apply_filters(results, filters)

        # Rank results
        results = self._rank_results(results, query)

        # Limit results
        results = results[:top_k]

        # Cache results
        self.result_cache[cache_key] = results

        # Record search history
        self._record_search(query, len(results), time.time() - start_time, filters)

        return results

    def _process_query(self, query: str) -> str:
        """Process and normalize query."""
        # Remove extra whitespace
        query = re.sub(r"\s+", " ", query.strip())

        # Convert to lowercase
        query = query.lower()

        # Remove special characters
        query = re.sub(r"[^\w\s.]", " ", query)

        return query

    def _get_query_embedding(self, query: str) -> np.ndarray:
        """Get embedding for query."""
        if TRANSFORMERS_AVAILABLE and self.embedding_model is not None:
            try:
                return self.embedding_model.encode([query])[0]
            except Exception:
                pass

        # Fallback: use word frequency
        words = re.findall(r"\w+", query)
        return np.array([hash(word) % 1000 for word in words[:50]])

    def _calculate_similarity(
        self, query_vec: np.ndarray, doc_vec: np.ndarray
    ) -> float:
        """Calculate similarity between query and document."""
        if len(query_vec) == 0 or len(doc_vec) == 0:
            return 0.0

        try:
            if isinstance(query_vec[0], (int, float)) and isinstance(
                doc_vec[0], (int, float)
            ):
                # Numeric vectors: cosine similarity
                norm_q = np.linalg.norm(query_vec)
                norm_d = np.linalg.norm(doc_vec)
                if norm_q == 0 or norm_d == 0:
                    return 0.0
                return np.dot(query_vec, doc_vec) / (norm_q * norm_d)
            elif isinstance(query_vec[0], str) and isinstance(doc_vec[0], str):
                # String vectors: overlap similarity
                query_set = set(query_vec)
                doc_set = set(doc_vec)
                if not query_set or not doc_set:
                    return 0.0
                return len(query_set & doc_set) / len(query_set | doc_set)
        except Exception:
            pass

        return 0.0

    def _extract_matched_terms(self, query: str, text: str) -> List[str]:
        """Extract terms from query that match the text."""
        if not query or not text:
            return []

        query_words = set(re.findall(r"\w+", query.lower()))
        text_words = set(re.findall(r"\w+", text.lower()))

        matched = list(query_words & text_words)
        return matched[:10]

    def _extract_context(self, text: str, query: str, window: int = 50) -> str:
        """Extract context around matched terms."""
        if not text or not query:
            return ""

        # Find first occurrence of any query word
        query_words = set(re.findall(r"\w+", query.lower()))
        words = re.findall(r"\w+", text.lower())

        for i, word in enumerate(words):
            if word in query_words:
                start = max(0, i - 10)
                end = min(len(words), i + 11)
                return " ".join(words[start:end])

        return text[:100] + "..." if len(text) > 100 else text

    def _apply_filters(
        self, results: List[SearchResult], filters: Dict
    ) -> List[SearchResult]:
        """Apply filters to search results."""
        filtered = results

        # Filter by document type
        if "document_type" in filters:
            filtered = [r for r in filtered if filters["document_type"] in r.document]

        # Filter by date (would need document dates)
        if "date_range" in filters:
            # Placeholder: would filter by document date
            pass

        # Filter by similarity threshold
        if "min_similarity" in filters:
            filtered = [
                r for r in filtered if r.similarity_score >= filters["min_similarity"]
            ]

        return filtered

    def _rank_results(
        self, results: List[SearchResult], query: str
    ) -> List[SearchResult]:
        """Rank search results by relevance."""
        if not results:
            return results

        # Compute relevance scores
        for result in results:
            # Base score: similarity
            score = result.similarity_score

            # Boost: matched terms count
            if result.matched_terms:
                score += len(result.matched_terms) * 0.01

            # Boost: term frequency in document
            if result.chunk_text:
                query_words = set(re.findall(r"\w+", query.lower()))
                text_words = set(re.findall(r"\w+", result.chunk_text.lower()))
                overlap = len(query_words & text_words)
                if overlap > 0:
                    score += 0.05 * (overlap / len(query_words))

            result.relevance_score = min(score, 1.0)

        # Sort by relevance
        results.sort(key=lambda x: x.relevance_score, reverse=True)

        return results

    def _record_search(
        self,
        query: str,
        results_count: int,
        execution_time: float,
        filters: Dict = None,
    ):
        """Record search in history."""
        search_query = SearchQuery(
            id=f"search_{int(time.time())}",
            query_text=query,
            query_type="semantic",
            timestamp=time.time(),
            user=st.session_state.get("username", "anonymous"),
            results_count=results_count,
            execution_time=execution_time,
            filters=filters or {},
        )
        self.search_history.append(search_query)

        # Keep last 1000 searches
        if len(self.search_history) > 1000:
            self.search_history = self.search_history[-1000:]

    def get_search_suggestions(self, partial_query: str) -> List[str]:
        """Get search suggestions based on partial query."""
        if not partial_query:
            return []

        partial = partial_query.lower()
        suggestions = []

        # Get from query history
        for search in self.search_history:
            if partial in search.query_text.lower():
                suggestions.append(search.query_text)

        # Get from common patterns
        if self.pattern_cache:
            for pattern_name, pattern in self.pattern_cache.items():
                if partial in pattern_name.lower():
                    suggestions.append(pattern_name)

        # Limit and deduplicate
        suggestions = list(set(suggestions))[:10]

        return suggestions

    def detect_patterns(self, min_frequency: int = 2) -> Dict[str, SearchPattern]:
        """
        Detect search patterns from history.

        Returns:
            Dict[str, SearchPattern]: Detected patterns
        """
        if len(self.search_history) < 5:
            return {}

        patterns = {}

        # Analyze query frequency
        query_counter = Counter([s.query_text for s in self.search_history])

        for query, count in query_counter.items():
            if count >= min_frequency:
                # Extract keywords
                keywords = re.findall(r"\w+", query)

                # Find documents affected
                affected_docs = []
                for search in self.search_history:
                    if search.query_text == query:
                        # Would need to get actual results
                        pass

                pattern = SearchPattern(
                    pattern_type="keyword" if len(keywords) < 5 else "semantic",
                    keywords=keywords[:5],
                    documents_affected=affected_docs,
                    similarity_range=(0.0, 0.0),
                    frequency=count,
                    last_detected=time.time(),
                )
                patterns[query] = pattern

        self.pattern_cache.update(patterns)
        return patterns

    def get_search_stats(self) -> Dict[str, Any]:
        """Get search statistics."""
        if not self.search_history:
            return {
                "total_searches": 0,
                "avg_results": 0,
                "avg_time": 0,
                "unique_queries": 0,
                "popular_queries": [],
            }

        queries = [s.query_text for s in self.search_history]
        unique_queries = len(set(queries))
        avg_results = sum(s.results_count for s in self.search_history) / len(
            self.search_history
        )
        avg_time = sum(s.execution_time for s in self.search_history) / len(
            self.search_history
        )

        # Popular queries
        query_counter = Counter(queries)
        popular = query_counter.most_common(5)

        return {
            "total_searches": len(self.search_history),
            "avg_results": avg_results,
            "avg_time": avg_time,
            "unique_queries": unique_queries,
            "popular_queries": popular,
            "cache_size": len(self.result_cache),
        }


# ==============================================================================
# UI COMPONENTS
# ==============================================================================


def render_smart_search_ui():
    """Render smart search UI."""
    st.subheader("🔍 Smart Search Engine")

    # Initialize search engine
    if "smart_search_engine" not in st.session_state:
        st.session_state.smart_search_engine = SmartSearchEngine()

    search_engine = st.session_state.smart_search_engine

    # Search input
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input(
            "Search for patterns across all documents",
            placeholder="E.g., 'similar conclusions about climate change' or 'duplicate methodology sections'",
            key="smart_search_input",
        )

    with col2:
        search_type = st.selectbox(  # noqa: F841
            "Type", ["Semantic", "Keyword", "Hybrid"], index=0
        )

    # Filters
    with st.expander("🔧 Filters", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            min_similarity = st.slider("Min Similarity", 0.0, 1.0, 0.3, 0.05)
        with col2:
            max_results = st.number_input("Max Results", 5, 50, 10)
        with col3:
            document_filter = st.multiselect(
                "Filter Documents", options=st.session_state.get("doc_names", [])
            )

    # Search button
    if st.button("🔍 Search", type="primary", use_container_width=True):
        if query.strip():
            with st.spinner("Searching..."):
                filters = {"min_similarity": min_similarity}
                if document_filter:
                    filters["document_type"] = "|".join(document_filter)

                results = search_engine.search(
                    query, top_k=max_results, filters=filters
                )

                st.session_state.last_search_results = results
                st.session_state.last_query = query

                st.success(f"Found {len(results)} results")
        else:
            st.warning("Please enter a search query")

    # Display results
    if (
        hasattr(st.session_state, "last_search_results")
        and st.session_state.last_search_results
    ):
        results = st.session_state.last_search_results

        st.markdown("---")
        st.markdown(f"### 📊 Results for: '{st.session_state.last_query}'")

        # Statistics
        col1, col2, col3 = st.columns(3)
        col1.metric("Results", len(results))
        if results:
            col2.metric(
                "Max Similarity", f"{max(r.similarity_score for r in results):.2%}"
            )
            col3.metric(
                "Avg Similarity",
                f"{sum(r.similarity_score for r in results) / len(results):.2%}",
            )

        # Display results
        for idx, result in enumerate(results, 1):
            with st.expander(
                f"{idx}. 📄 {result.document} - {result.similarity_score:.2%} similarity",
                expanded=idx == 1,
            ):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown("**Matched Text:**")
                    st.info(
                        result.chunk_text[:500] + "..."
                        if len(result.chunk_text) > 500
                        else result.chunk_text
                    )

                with col2:
                    st.markdown("**Details:**")
                    st.caption(f"Chunk: #{result.chunk_index}")
                    if result.matched_terms:
                        st.caption(
                            f"Matched Terms: {', '.join(result.matched_terms[:5])}"
                        )
                    st.caption(f"Relevance: {result.relevance_score:.2%}")

                if result.context:
                    st.markdown("**Context:**")
                    st.markdown(f"`{result.context}`")

    # Search Suggestions
    if query:
        suggestions = search_engine.get_search_suggestions(query)
        if suggestions:
            st.markdown("#### 💡 Suggested Searches")
            cols = st.columns(min(4, len(suggestions)))
            for col, suggestion in zip(cols, suggestions[:4]):
                if col.button(suggestion, use_container_width=True):
                    st.session_state.smart_search_input = suggestion
                    st.rerun()

    # Search History
    st.markdown("---")
    with st.expander("📜 Search History", expanded=False):
        if search_engine.search_history:
            df = pd.DataFrame(
                [
                    {
                        "Query": s.query_text[:50],
                        "Results": s.results_count,
                        "Time": f"{s.execution_time * 1000:.1f}ms",
                        "Timestamp": datetime.fromtimestamp(s.timestamp).strftime(
                            "%H:%M:%S"
                        ),
                    }
                    for s in search_engine.search_history[-20:]
                ]
            )
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No search history yet")


def render_search_analytics():
    """Render search analytics dashboard."""
    st.subheader("📊 Search Analytics")

    if "smart_search_engine" not in st.session_state:
        st.info("No search data available")
        return

    search_engine = st.session_state.smart_search_engine
    stats = search_engine.get_search_stats()

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Searches", stats["total_searches"])
    col2.metric("Unique Queries", stats["unique_queries"])
    col3.metric("Avg Results", f"{stats['avg_results']:.1f}")
    col4.metric("Avg Time", f"{stats['avg_time'] * 1000:.1f}ms")

    # Popular queries
    if stats["popular_queries"]:
        st.markdown("#### 🔥 Popular Queries")
        df_popular = pd.DataFrame(
            stats["popular_queries"], columns=["Query", "Frequency"]
        )
        st.dataframe(df_popular, use_container_width=True, hide_index=True)

    # Pattern detection
    if st.button("🔄 Detect Search Patterns", use_container_width=True):
        patterns = search_engine.detect_patterns()
        if patterns:
            st.success(f"Found {len(patterns)} search patterns")
            for pattern_name, pattern in patterns.items():
                st.markdown(f"**{pattern_name}**")
                st.caption(
                    f"Frequency: {pattern.frequency} | Type: {pattern.pattern_type}"
                )
        else:
            st.info("No patterns detected yet")


# ==============================================================================
# INITIALIZATION
# ==============================================================================


def initialize_smart_search():
    """Initialize smart search engine."""
    if "smart_search_initialized" not in st.session_state:
        st.session_state.smart_search_initialized = True

        # Initialize search engine
        engine = SmartSearchEngine()
        st.session_state.smart_search_engine = engine

        # Index existing documents if available
        if "doc_texts" in st.session_state and "chunked_docs" in st.session_state:
            engine.index_documents(
                st.session_state.doc_texts, st.session_state.chunked_docs
            )
