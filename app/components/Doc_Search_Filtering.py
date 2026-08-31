# ───────────────────────────────────────────────────────────────────────────────
# ── SECTION: ADVANCED DOCUMENT SEARCH & SMART FILTERING (Issue #1987) ──────
# ───────────────────────────────────────────────────────────────────────────────

import re
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# ── Search Data Models ─────────────────────────────────────────────────────


@dataclass
class SearchResult:
    """Represents a single search result"""

    document_name: str
    score: float
    snippet: str
    match_type: str  # 'full_text', 'semantic', 'metadata', 'tag'
    metadata: Dict[str, Any] = None
    highlights: List[Tuple[int, int]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.highlights is None:
            self.highlights = []

    def to_dict(self) -> Dict:
        return {
            "document_name": self.document_name,
            "score": self.score,
            "snippet": self.snippet,
            "match_type": self.match_type,
            "metadata": self.metadata,
            "highlights": self.highlights,
        }


@dataclass
class SearchQuery:
    """Represents a search query with filters"""

    id: str
    query_text: str
    filters: dict[str, Any]
    search_type: str  # 'full_text', 'semantic', 'hybrid'
    timestamp: datetime
    user_id: str
    results_count: int = 0
    is_saved: bool = False
    name: str = None

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "query_text": self.query_text,
            "filters": self.filters,
            "search_type": self.search_type,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "results_count": self.results_count,
            "is_saved": self.is_saved,
            "name": self.name,
        }


@dataclass
class SearchFilter:
    """Represents a search filter"""

    field: str
    operator: str  # 'eq', 'ne', 'gt', 'lt', 'gte', 'lte', 'contains', 'in'
    value: Any

    def matches(self, document: Dict) -> bool:
        """Check if document matches this filter"""
        doc_value = document.get(self.field)

        if self.operator == "eq":
            return doc_value == self.value
        elif self.operator == "ne":
            return doc_value != self.value
        elif self.operator == "gt":
            return doc_value > self.value
        elif self.operator == "lt":
            return doc_value < self.value
        elif self.operator == "gte":
            return doc_value >= self.value
        elif self.operator == "lte":
            return doc_value <= self.value
        elif self.operator == "contains":
            return self.value.lower() in str(doc_value).lower()
        elif self.operator == "in":
            return doc_value in self.value
        return False


# ── Search Engine ──────────────────────────────────────────────────────────


class SearchEngine:
    """Advanced search engine with multiple search strategies"""

    def __init__(self):
        self.document_index = {}
        self.metadata_index = {}
        self.full_text_index = {}
        self.semantic_embeddings = {}
        self.search_history = []
        self.saved_searches = {}
        self.tag_index = defaultdict(set)

    def index_document(
        self,
        doc_name: str,
        content: str,
        metadata: Dict = None,
        embeddings: np.ndarray = None,
        tags: List[str] = None,
    ):
        """Index a document for search"""
        # Store full text
        self.document_index[doc_name] = content
        self.metadata_index[doc_name] = metadata or {}

        # Index words for full-text search
        words = self._tokenize(content)
        for word in words:
            if word not in self.full_text_index:
                self.full_text_index[word] = set()
            self.full_text_index[word].add(doc_name)

        # Store embeddings for semantic search
        if embeddings is not None:
            self.semantic_embeddings[doc_name] = embeddings

        # Index tags
        if tags:
            for tag in tags:
                self.tag_index[tag].add(doc_name)

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words for indexing"""
        # Simple tokenization - can be improved
        text = text.lower()
        words = re.findall(r"\b[a-z0-9]+\b", text)
        # Remove common stopwords
        stopwords = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "without",
            "by",
            "from",
            "up",
            "down",
        }
        return [w for w in words if w not in stopwords and len(w) > 2]

    def search_full_text(
        self, query: str, filters: List[SearchFilter] = None, max_results: int = 50
    ) -> List[SearchResult]:
        """Perform full-text search"""
        query_words = self._tokenize(query)
        if not query_words:
            return []

        # Find documents containing query words
        doc_scores = {}
        for word in query_words:
            if word in self.full_text_index:
                for doc in self.full_text_index[word]:
                    doc_scores[doc] = doc_scores.get(doc, 0) + 1

        # Sort by score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        # Apply filters
        results = []
        for doc_name, score in sorted_docs[:max_results]:
            # Apply filters
            if filters and not self._apply_filters(doc_name, filters):
                continue

            # Generate snippet
            content = self.document_index.get(doc_name, "")
            snippet = self._generate_snippet(content, query_words)

            # Create result
            result = SearchResult(
                document_name=doc_name,
                score=score / len(query_words) if query_words else 0,
                snippet=snippet,
                match_type="full_text",
                metadata=self.metadata_index.get(doc_name, {}),
            )
            results.append(result)

        return results

    def search_semantic(
        self,
        query_embedding: np.ndarray,
        filters: List[SearchFilter] = None,
        threshold: float = 0.5,
        max_results: int = 50,
    ) -> List[SearchResult]:
        """Perform semantic search using embeddings"""
        if not self.semantic_embeddings or query_embedding is None:
            return []

        results = []
        for doc_name, doc_embedding in self.semantic_embeddings.items():
            # Calculate similarity
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )

            if similarity >= threshold:
                # Apply filters
                if filters and not self._apply_filters(doc_name, filters):
                    continue

                content = self.document_index.get(doc_name, "")
                result = SearchResult(
                    document_name=doc_name,
                    score=float(similarity),
                    snippet=content[:300] + "..." if len(content) > 300 else content,
                    match_type="semantic",
                    metadata=self.metadata_index.get(doc_name, {}),
                )
                results.append(result)

        # Sort by similarity
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:max_results]

    def search_hybrid(
        self,
        query: str,
        query_embedding: np.ndarray = None,
        filters: List[SearchFilter] = None,
        semantic_weight: float = 0.5,
        max_results: int = 50,
    ) -> List[SearchResult]:
        """Perform hybrid search combining full-text and semantic"""
        # Get full-text results
        text_results = self.search_full_text(query, filters, max_results)

        # Get semantic results if embedding available
        semantic_results = []
        if query_embedding is not None:
            semantic_results = self.search_semantic(
                query_embedding, filters, max_results
            )

        # Combine results
        combined = {}

        # Add full-text results
        for result in text_results:
            combined[result.document_name] = {
                "result": result,
                "text_score": result.score,
                "semantic_score": 0,
                "count": 1,
            }

        # Add semantic results
        for result in semantic_results:
            if result.document_name in combined:
                combined[result.document_name]["semantic_score"] = result.score
                combined[result.document_name]["count"] += 1
            else:
                combined[result.document_name] = {
                    "result": result,
                    "text_score": 0,
                    "semantic_score": result.score,
                    "count": 1,
                }

        # Calculate hybrid scores
        hybrid_results = []
        for doc_name, data in combined.items():
            if data["count"] == 1:
                # Only one type of match
                score = data["text_score"] or data["semantic_score"]
            else:
                # Hybrid score
                score = (
                    semantic_weight * data["semantic_score"]
                    + (1 - semantic_weight) * data["text_score"]
                )

            result = data["result"]
            result.score = score
            result.match_type = "hybrid"
            hybrid_results.append(result)

        # Sort by score
        hybrid_results.sort(key=lambda x: x.score, reverse=True)
        return hybrid_results[:max_results]

    def _apply_filters(self, doc_name: str, filters: List[SearchFilter]) -> bool:
        """Apply filters to a document"""
        metadata = self.metadata_index.get(doc_name, {})
        for filter_obj in filters:
            if not filter_obj.matches(metadata):
                return False
        return True

    def _generate_snippet(
        self, content: str, query_words: List[str], context_chars: int = 100
    ) -> str:
        """Generate a snippet with context around matches"""
        if not content or not query_words:
            return content[:300] + "..." if len(content) > 300 else content

        # Find first match position
        content_lower = content.lower()
        best_pos = -1
        for word in query_words:
            pos = content_lower.find(word.lower())
            if pos != -1 and (best_pos == -1 or pos < best_pos):
                best_pos = pos

        if best_pos == -1:
            return content[:300] + "..." if len(content) > 300 else content

        # Extract context
        start = max(0, best_pos - context_chars)
        end = min(len(content), best_pos + context_chars + 100)
        snippet = content[start:end]

        # Add ellipsis
        if start > 0:
            snippet = "..." + snippet
        if end < len(content):
            snippet = snippet + "..."

        return snippet

    def get_search_history(self) -> List[SearchQuery]:
        """Get search history"""
        return self.search_history

    def save_search(self, query: SearchQuery, name: str) -> str:
        """Save a search query"""
        query.is_saved = True
        query.name = name
        search_id = str(uuid.uuid4())
        self.saved_searches[search_id] = query
        return search_id

    def get_saved_searches(self) -> Dict[str, SearchQuery]:
        """Get saved searches"""
        return self.saved_searches

    def delete_saved_search(self, search_id: str) -> bool:
        """Delete a saved search"""
        if search_id in self.saved_searches:
            del self.saved_searches[search_id]
            return True
        return False

    def get_search_stats(self) -> Dict:
        """Get search statistics"""
        return {
            "total_searches": len(self.search_history),
            "saved_searches": len(self.saved_searches),
            "indexed_documents": len(self.document_index),
            "unique_tags": len(self.tag_index),
        }


# ── Advanced Search Parser ─────────────────────────────────────────────────


class SearchQueryParser:
    """Parse search queries with advanced syntax"""

    def __init__(self):
        self.operators = ["AND", "OR", "NOT"]
        self.field_pattern = re.compile(r'(\w+):(".*?"|\S+)')

    def parse(self, query: str) -> Dict:
        """Parse a search query into components"""
        result = {
            "keywords": [],
            "exact_phrases": [],
            "excluded": [],
            "fields": {},
            "operators": [],
        }

        # Extract field searches
        for match in self.field_pattern.finditer(query):
            field = match.group(1)
            value = match.group(2).strip('"')
            result["fields"][field] = value
            query = query.replace(match.group(0), "")

        # Parse remaining text
        parts = query.split()
        i = 0
        while i < len(parts):
            token = parts[i]

            # Check for operators
            if token.upper() in self.operators:
                result["operators"].append(token.upper())
                i += 1
                continue

            # Check for exact phrase
            if token.startswith('"'):
                phrase = token
                i += 1
                while i < len(parts) and not parts[i].endswith('"'):
                    phrase += " " + parts[i]
                    i += 1
                if i < len(parts) and parts[i].endswith('"'):
                    phrase += " " + parts[i]
                    i += 1
                result["exact_phrases"].append(phrase.strip('"'))
                continue

            # Check for excluded term
            if token.startswith("-"):
                result["excluded"].append(token[1:])
                i += 1
                continue

            # Regular keyword
            result["keywords"].append(token)
            i += 1

        return result


# ── Smart Filter Builder ────────────────────────────────────────────────────


class SmartFilterBuilder:
    """Build and manage search filters"""

    def __init__(self):
        self.available_fields = {
            "date": ["eq", "gt", "lt", "gte", "lte"],
            "size": ["gt", "lt", "gte", "lte", "eq"],
            "similarity": ["gt", "lt", "gte", "lte"],
            "author": ["eq", "contains"],
            "type": ["eq", "in"],
            "tags": ["contains", "in"],
            "status": ["eq"],
            "word_count": ["gt", "lt", "gte", "lte"],
        }

    def create_filter(self, field: str, operator: str, value: Any) -> SearchFilter:
        """Create a new search filter"""
        if field not in self.available_fields:
            raise ValueError(f"Unknown field: {field}")
        if operator not in self.available_fields[field]:
            raise ValueError(f"Invalid operator for {field}: {operator}")
        return SearchFilter(field, operator, value)

    def build_from_dict(self, filter_dict: Dict) -> List[SearchFilter]:
        """Build filters from dictionary"""
        filters = []
        for field, conditions in filter_dict.items():
            if field in self.available_fields:
                for op, value in conditions.items():
                    if op in self.available_fields[field]:
                        filters.append(SearchFilter(field, op, value))
        return filters

    def get_filter_options(self) -> Dict:
        """Get available filter options"""
        return self.available_fields


# ── Search UI Components ───────────────────────────────────────────────────


def render_search_bar(search_engine: SearchEngine):
    """Render the main search bar"""
    st.subheader("🔍 Advanced Search")

    # Search input
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        query = st.text_input(
            "Search query:",
            placeholder="Enter search terms... (use AND, OR, NOT, field:value)",
            key="search_query_input",
        )

    with col2:
        search_type = st.selectbox(
            "Search type:",
            ["full_text", "semantic", "hybrid"],
            key="search_type_select",
        )

    with col3:
        if st.button("🔍 Search", type="primary", key="search_button"):
            st.session_state["active_search"] = True
            st.session_state["search_query"] = query
            st.session_state["search_type"] = search_type
            st.rerun()

    # Search tips
    with st.expander("ℹ️ Search Tips", expanded=False):
        st.markdown("""
        **Basic Search:**
        - Use `AND`, `OR`, `NOT` between terms: `plagiarism AND detection`
        - Use quotes for exact phrases: `"semantic similarity"`
        - Exclude terms with `-`: `-citations`
        
        **Field Search:**
        - `author:"John Doe"` - Search by author
        - `date:>2024-01-01` - Search by date
        - `similarity:>0.7` - Search by similarity score
        - `tags:research` - Search by tags
        
        **Examples:**
        - `"machine learning" OR "deep learning"`
        - `author:Smith AND date:>2024-01-01`
        - `plagiarism -self NOT draft`
        """)

    return query, search_type


def render_search_filters(filter_builder: SmartFilterBuilder):
    """Render search filters UI"""
    st.subheader("🎯 Filters")

    filters = []

    # Date filter
    col1, col2 = st.columns(2)
    with col1:
        date_filter = st.selectbox(
            "Date range:",
            ["None", "Today", "Last 7 Days", "Last 30 Days", "Last 90 Days"],
            key="date_filter_select",
        )

    with col2:
        if date_filter != "None":
            date_operator = st.selectbox(
                "Date operator:", ["gt", "lt", "gte", "lte", "eq"], key="date_op_select"
            )

    if date_filter != "None":
        today = datetime.now()
        if date_filter == "Today":
            date_value = today.strftime("%Y-%m-%d")
        elif date_filter == "Last 7 Days":
            date_value = (today - timedelta(days=7)).strftime("%Y-%m-%d")
        elif date_filter == "Last 30 Days":
            date_value = (today - timedelta(days=30)).strftime("%Y-%m-%d")
        elif date_filter == "Last 90 Days":
            date_value = (today - timedelta(days=90)).strftime("%Y-%m-%d")

        filters.append(SearchFilter("date", date_operator, date_value))

    # Similarity filter
    col1, col2 = st.columns(2)
    with col1:
        use_similarity_filter = st.checkbox("Similarity filter", key="sim_filter_check")

    if use_similarity_filter:
        with col2:
            sim_operator = st.selectbox(
                "Operator:", ["gt", "lt", "gte", "lte"], key="sim_op_select"
            )

        sim_value = st.slider(
            "Similarity threshold:", 0.0, 1.0, 0.5, 0.05, key="sim_value_slider"
        )
        filters.append(SearchFilter("similarity", sim_operator, sim_value))

    # Author filter
    author = st.text_input(
        "Author:", placeholder="Search by author name...", key="author_filter_input"
    )
    if author:
        author_operator = st.selectbox(
            "Author operator:", ["eq", "contains"], key="author_op_select"
        )
        filters.append(SearchFilter("author", author_operator, author))

    # Tags filter
    tags = st.multiselect(
        "Tags:",
        ["research", "draft", "final", "review", "plagiarism", "original"],
        key="tags_filter_select",
    )
    if tags:
        filters.append(SearchFilter("tags", "in", tags))

    # Reset filters
    if st.button("🔄 Reset Filters", key="reset_filters_button"):
        st.session_state["search_filters"] = []
        st.rerun()

    return filters


def render_search_results(results: List[SearchResult]):
    """Render search results"""
    st.subheader(f"📋 Results ({len(results)})")

    if not results:
        st.info("No results found. Try adjusting your search query or filters.")
        return

    # Results per page
    page_size = 10
    total_pages = (len(results) + page_size - 1) // page_size

    if "search_page" not in st.session_state:
        st.session_state["search_page"] = 0

    col1, col2, col3 = st.columns([3, 1, 1])
    with col2:
        st.write(f"Page {st.session_state['search_page'] + 1} of {total_pages}")
    with col3:
        if st.button(
            "Next →", disabled=st.session_state["search_page"] >= total_pages - 1
        ):
            st.session_state["search_page"] += 1
            st.rerun()

    start_idx = st.session_state["search_page"] * page_size
    end_idx = min(start_idx + page_size, len(results))

    for idx in range(start_idx, end_idx):
        result = results[idx]

        # Determine color based on match type
        colors = {
            "full_text": "#4CAF50",
            "semantic": "#2196F3",
            "hybrid": "#FF9800",
            "metadata": "#9C27B0",
            "tag": "#F44336",
        }
        color = colors.get(result.match_type, "#666666")

        with st.expander(
            f"#{idx + 1} - {result.document_name} "
            f"[{result.score * 100:.1f}%] {result.match_type.upper()}",
            expanded=(idx == start_idx),
        ):
            # Score and type
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Score", f"{result.score * 100:.1f}%")
            with col2:
                st.metric("Type", result.match_type.title())
            with col3:
                st.metric("Matches", len(result.highlights) if result.highlights else 0)
            with col4:
                st.metric("Status", "✅" if result.score > 0.7 else "🔄 Review")

            # Snippet
            st.markdown("**Snippet:**")
            st.markdown(
                f"<div style='background:#f5f5f5;padding:10px;border-radius:5px;'>{result.snippet}</div>",
                unsafe_allow_html=True,
            )

            # Metadata
            if result.metadata:
                with st.expander("📋 Metadata", expanded=False):
                    st.json(result.metadata)

            # Actions
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button(
                    "📄 View Document", key=f"view_{result.document_name}_{idx}"
                ):
                    st.session_state["selected_doc"] = result.document_name
                    st.rerun()
            with col2:
                if st.button(
                    "📝 Annotate", key=f"annotate_{result.document_name}_{idx}"
                ):
                    st.session_state["annotate_doc"] = result.document_name
                    st.rerun()
            with col3:
                if st.button("📊 Compare", key=f"compare_{result.document_name}_{idx}"):
                    st.session_state["compare_doc"] = result.document_name
                    st.rerun()


def render_saved_searches(search_engine: SearchEngine):
    """Render saved searches UI"""
    st.subheader("⭐ Saved Searches")

    saved = search_engine.get_saved_searches()
    if not saved:
        st.info("No saved searches.")
        return

    for search_id, query in saved.items():
        with st.expander(f"{query.name} ({query.search_type})", expanded=False):
            st.markdown(f"**Query:** {query.query_text}")
            st.markdown(f"**Created:** {query.timestamp.strftime('%Y-%m-%d %H:%M')}")
            st.markdown(f"**Results:** {query.results_count}")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔍 Run", key=f"run_saved_{search_id}"):
                    st.session_state["search_query"] = query.query_text
                    st.session_state["active_search"] = True
                    st.rerun()
            with col2:
                if st.button("🗑️ Delete", key=f"delete_saved_{search_id}"):
                    search_engine.delete_saved_search(search_id)
                    st.rerun()


def render_search_analytics(search_engine: SearchEngine):
    """Render search analytics dashboard"""
    st.subheader("📊 Search Analytics")

    stats = search_engine.get_search_stats()

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Searches", stats["total_searches"])
    col2.metric("Saved Searches", stats["saved_searches"])
    col3.metric("Indexed Documents", stats["indexed_documents"])
    col4.metric("Unique Tags", stats["unique_tags"])

    # Search history
    if search_engine.search_history:
        st.subheader("🕐 Search History")
        history_data = []
        for query in search_engine.search_history[-20:]:
            history_data.append(
                {
                    "Query": query.query_text[:50] + "..."
                    if len(query.query_text) > 50
                    else query.query_text,
                    "Type": query.search_type,
                    "Results": query.results_count,
                    "Timestamp": query.timestamp.strftime("%Y-%m-%d %H:%M"),
                }
            )

        df = pd.DataFrame(history_data)
        st.dataframe(df, use_container_width=True)


def render_advanced_search_dashboard():
    """Render the complete search dashboard"""
    # Initialize search engine
    if "search_engine" not in st.session_state:
        st.session_state["search_engine"] = SearchEngine()
        st.session_state["filter_builder"] = SmartFilterBuilder()
        st.session_state["search_page"] = 0
        st.session_state["search_results"] = []
        st.session_state["active_search"] = False

    search_engine = st.session_state["search_engine"]
    filter_builder = st.session_state["filter_builder"]

    # Search header
    query, search_type = render_search_bar(search_engine)
    filters = render_search_filters(filter_builder)

    # Execute search
    if st.session_state.get("active_search", False) or (
        query and st.session_state.get("search_query")
    ):
        search_query = st.session_state.get("search_query", query)
        search_type = st.session_state.get("search_type", search_type)

        with st.spinner("Searching..."):
            if search_type == "full_text":
                results = search_engine.search_full_text(search_query, filters)
            elif search_type == "semantic":
                # Use a dummy embedding for demo (should use actual embeddings)
                query_embedding = np.random.randn(384)
                results = search_engine.search_semantic(query_embedding, filters)
            else:  # hybrid
                query_embedding = np.random.randn(384)  # Dummy embedding
                results = search_engine.search_hybrid(
                    search_query, query_embedding, filters
                )

            st.session_state["search_results"] = results

            # Save to history
            if results:
                search_query_obj = SearchQuery(
                    id=str(uuid.uuid4()),
                    query_text=search_query,
                    filters=[f.to_dict() for f in filters],
                    search_type=search_type,
                    timestamp=datetime.now(),
                    user_id=st.session_state.get("user_id", "unknown"),
                    results_count=len(results),
                )
                search_engine.search_history.append(search_query_obj)

            st.session_state["active_search"] = False

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📋 Results", "⭐ Saved Searches", "📊 Analytics"])

    with tab1:
        render_search_results(st.session_state.get("search_results", []))

    with tab2:
        render_saved_searches(search_engine)

    with tab3:
        render_search_analytics(search_engine)


# ── Integration with Main App ─────────────────────────────────────────────


def integrate_search_system():
    """Initialize and integrate search system with main app"""
    initialize_review_system()

    # Add search tab to main tabs
    st.subheader("🔍 Search & Filter System")
    render_advanced_search_dashboard()


# ── End of Search System ──────────────────────────────────────────────────
# ───────────────────────────────────────────────────────────────────────────────
