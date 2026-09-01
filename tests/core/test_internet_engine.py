import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

from src.core.internet_engine.models import SearchQuery, FetchedSource, TransientIndexConfig
from src.core.internet_engine.scraper import QueryGenerator, HtmlCleaner, MockSearchProvider, AsyncInternetFetcher
from src.core.internet_engine.transient_index import TransientInternetIndex
from src.core.internet_engine.engine import InternetPlagiarismEngine

# --- Tests for QueryGenerator ---

def test_query_generator_basic():
    generator = QueryGenerator(max_queries=2, max_words_per_query=5)
    # A text with long sentences
    text = (
        "The quick brown fox jumps over the lazy dog in the forest. "
        "Artificial intelligence is rapidly transforming the modern technological landscape across the globe. "
        "Short sentence. "
        "Another very long and complex sentence that should definitely be selected as a search query for testing."
    )
    queries = generator.generate_queries("doc_1", text)
    
    assert len(queries) == 2
    # Ensure they are sorted by length (descending)
    assert "Another very long" in queries[0].query_text
    assert "Artificial intelligence" in queries[1].query_text
    
def test_query_generator_empty():
    generator = QueryGenerator()
    queries = generator.generate_queries("doc_1", "")
    assert len(queries) == 0

# --- Tests for HtmlCleaner ---

def test_html_cleaner_valid():
    html = """
    <html>
        <head><title>Test Title</title><script>alert('x');</script></head>
        <body>
            <nav>Menu</nav>
            <header>Header</header>
            <main>
                <h1>Main Heading</h1>
                <p>This is the important content we want to keep.</p>
                <!-- This is a comment -->
            </main>
            <aside>Sidebar</aside>
            <footer>Footer</footer>
        </body>
    </html>
    """
    clean_text = HtmlCleaner.clean_html(html, "http://example.com")
    assert "This is the important content we want to keep." in clean_text
    assert "Main Heading" in clean_text
    assert "alert('x')" not in clean_text
    assert "Menu" not in clean_text
    assert "Footer" not in clean_text
    assert "This is a comment" not in clean_text

def test_html_cleaner_empty():
    clean_text = HtmlCleaner.clean_html("", "http://example.com")
    assert clean_text == ""

# --- Tests for AsyncInternetFetcher ---

@pytest.mark.asyncio
async def test_async_fetcher_success():
    fetcher = AsyncInternetFetcher(max_retries=1)
    
    # Mock robots.txt check to always allow
    with patch.object(fetcher, '_can_fetch', new_callable=AsyncMock) as mock_can_fetch:
        mock_can_fetch.return_value = True
        
        # Mock aiohttp session and response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text.return_value = "<html><body>Useful text content</body></html>"
        
        mock_session = AsyncMock()
        # Ensure session.get returns an async context manager
        mock_session.get.return_value.__aenter__.return_value = mock_response
        
        result = await fetcher.fetch_url(mock_session, "http://example.com/valid")
        
        assert result.is_valid is True
        assert result.url == "http://example.com/valid"
        assert "Useful text content" in result.content

@pytest.mark.asyncio
async def test_async_fetcher_blocked_by_robots():
    fetcher = AsyncInternetFetcher()
    with patch.object(fetcher, '_can_fetch', new_callable=AsyncMock) as mock_can_fetch:
        mock_can_fetch.return_value = False
        
        mock_session = AsyncMock()
        result = await fetcher.fetch_url(mock_session, "http://example.com/blocked")
        
        assert result.is_valid is False
        assert "Blocked by robots.txt" in result.error_message

# --- Tests for TransientInternetIndex ---

@patch('src.core.internet_engine.transient_index.embed_documents')
@patch('src.core.internet_engine.transient_index.faiss')
def test_transient_index_build(mock_faiss, mock_embed):
    # Mock FAISS
    mock_index = MagicMock()
    mock_faiss.IndexFlatIP.return_value = mock_index
    
    # Mock Embeddings
    import numpy as np
    mock_embed.return_value = {
        "http://example.com": np.array([[0.1] * 384])
    }
    
    config = TransientIndexConfig()
    index = TransientInternetIndex(config)
    
    sources = [
        FetchedSource(url="http://example.com", title="Test", content="Some valid long text content.", fetch_duration_ms=10.0, is_valid=True)
    ]
    
    index.build_index(sources)
    
    assert index.index is not None
    assert len(index.registry) == 1
    assert index.registry[0]["url"] == "http://example.com"
    mock_index.add.assert_called_once()

# --- Tests for InternetPlagiarismEngine Orchestrator ---

@pytest.mark.asyncio
async def test_engine_orchestrator():
    engine = InternetPlagiarismEngine()
    
    # We will patch the search and fetch steps to avoid real network calls
    with patch.object(engine.search_provider, 'search', new_callable=AsyncMock) as mock_search:
        # Return 1 fake result
        from src.core.internet_engine.models import SearchResult
        mock_search.return_value = [
            SearchResult(url="http://example.com/1", title="T1", snippet="S1", provider="mock", rank=1)
        ]
        
        with patch.object(engine.fetcher, 'fetch_all', new_callable=AsyncMock) as mock_fetch:
            # Return 1 fake fetched source
            mock_fetch.return_value = [
                FetchedSource(url="http://example.com/1", title="T1", content="This is the plagiarized text content.", fetch_duration_ms=50.0, is_valid=True)
            ]
            
            # Patch FAISS index to simulate a match without needing actual FAISS and embeddings
            with patch.object(TransientInternetIndex, 'search') as mock_index_search:
                from src.core.internet_engine.models import TransientMatch
                mock_index_search.return_value = [
                    TransientMatch(submitted_doc_id="doc_1", internet_url="http://example.com/1", similarity_score=0.95, matched_chunk_submitted="text", matched_chunk_internet="text")
                ]
                
                # We mock build_index to do nothing
                with patch.object(TransientInternetIndex, 'build_index'):
                    report = await engine.scan_document_async("doc_1", "This is the plagiarized text content.")
                    
                    assert report.submitted_doc_id == "doc_1"
                    assert report.total_queries_run > 0
                    assert report.total_urls_fetched == 1
                    assert len(report.matches) == 1
                    assert report.matches[0].similarity_score == 0.95
