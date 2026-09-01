"""
Data models and dataclasses for the Internet Scraping and Transient Indexing module.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime

@dataclass
class SearchQuery:
    """Represents a generated search query derived from a submitted document."""
    query_text: str
    source_document_id: str
    weight: float = 1.0

@dataclass
class SearchResult:
    """Represents a single URL returned by a search provider."""
    url: str
    title: str
    snippet: str
    provider: str
    rank: int

@dataclass
class FetchedSource:
    """Represents the cleaned text content successfully fetched from a SearchResult URL."""
    url: str
    title: str
    content: str
    fetch_duration_ms: float
    timestamp: datetime = field(default_factory=datetime.now)
    is_valid: bool = True
    error_message: Optional[str] = None

@dataclass
class TransientIndexConfig:
    """Configuration for building the transient FAISS index."""
    chunk_size: int = 500
    chunk_overlap: int = 50
    similarity_threshold: float = 0.70

@dataclass
class TransientMatch:
    """Represents a plagiarism match between a submitted document and an internet source."""
    submitted_doc_id: str
    internet_url: str
    similarity_score: float
    matched_chunk_submitted: str
    matched_chunk_internet: str

@dataclass
class InternetPlagiarismReport:
    """Final output of the Internet Engine comparison."""
    submitted_doc_id: str
    total_queries_run: int
    total_urls_fetched: int
    matches: List[TransientMatch] = field(default_factory=list)
    execution_time_ms: float = 0.0
    failed_urls: Dict[str, str] = field(default_factory=dict)
