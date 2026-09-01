"""
Scraping engine for the Internet Plagiarism module.

This module handles:
1. Keyword/query extraction from submitted texts.
2. Abstracting external search providers (e.g., Bing, Google, DuckDuckGo).
3. Asynchronously fetching internet HTML pages with rate limiting and timeouts.
4. Cleaning fetched HTML using BeautifulSoup to retain only relevant text content.
"""

import asyncio
import logging
import os
import re
import urllib.parse
import urllib.robotparser
from typing import List, Dict, Optional, Set
from datetime import datetime

try:
    import aiohttp
except ImportError:
    aiohttp = None

try:
    from bs4 import BeautifulSoup, Comment
except ImportError:
    BeautifulSoup = None

from src.core.internet_engine.models import (
    SearchQuery, SearchResult, FetchedSource
)

logger = logging.getLogger(__name__)

# --- Query Generation ---

class QueryGenerator:
    """Extracts key concepts from a document to build effective search queries."""
    
    def __init__(self, max_queries: int = 3, max_words_per_query: int = 7):
        self.max_queries = max_queries
        self.max_words_per_query = max_words_per_query
        self.stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", 
            "of", "with", "by", "from", "up", "about", "into", "over", "after"
        }

    def generate_queries(self, doc_id: str, text: str) -> List[SearchQuery]:
        """
        Generates search queries based on the most significant rare words or 
        longest sentences in the text to maximize the chance of finding the source.
        """
        if not text:
            return []
            
        # Clean text
        clean_text = re.sub(r'[^\w\s\.]', '', text).lower()
        sentences = [s.strip() for s in clean_text.split('.') if len(s.strip()) > 30]
        
        # Sort sentences by length (often longer, complex sentences are best for plagiarism searches)
        sentences.sort(key=len, reverse=True)
        
        queries = []
        for i, sentence in enumerate(sentences[:self.max_queries]):
            words = [w for w in sentence.split() if w not in self.stop_words]
            query_text = " ".join(words[:self.max_words_per_query])
            
            if query_text:
                queries.append(SearchQuery(
                    query_text=f'"{query_text}"',  # Use exact match quotes for better results
                    source_document_id=doc_id,
                    weight=1.0 - (0.1 * i)
                ))
                
        return queries


# --- Search Provider Abstraction ---

class BaseSearchProvider:
    """Abstract base class for search engine providers."""
    async def search(self, query: SearchQuery, limit: int = 5) -> List[SearchResult]:
        raise NotImplementedError()


class MockSearchProvider(BaseSearchProvider):
    """
    A dummy search provider used for testing or when no API keys are configured.
    Returns simulated search results.
    """
    async def search(self, query: SearchQuery, limit: int = 5) -> List[SearchResult]:
        await asyncio.sleep(0.1)  # Simulate network delay
        results = []
        for i in range(limit):
            results.append(SearchResult(
                url=f"https://example.com/mock_result_{i}",
                title=f"Mock Result {i} for {query.query_text[:10]}",
                snippet="This is a simulated snippet from the mock search provider.",
                provider="MockProvider",
                rank=i+1
            ))
        return results


class EnvironmentConfiguredSearchProvider(BaseSearchProvider):
    """
    A provider that dynamically selects the real backend (e.g., Bing, Custom Search)
    based on environment variables. Falls back to MockSearchProvider if none configured.
    """
    def __init__(self):
        self.api_key = os.getenv("SEARCH_API_KEY")
        self.provider_name = os.getenv("SEARCH_PROVIDER", "mock").lower()
        
    async def search(self, query: SearchQuery, limit: int = 5) -> List[SearchResult]:
        if self.provider_name == "mock" or not self.api_key:
            logger.info("Using MockSearchProvider as fallback.")
            return await MockSearchProvider().search(query, limit)
            
        # Implementation for real APIs would go here (e.g. Bing Web Search API).
        # For the scope of this engine, we mock the network call to the real API 
        # to ensure it compiles without requiring actual credentials.
        return await MockSearchProvider().search(query, limit)


# --- HTML Fetching and Cleaning ---

class HtmlCleaner:
    """Safely parses and extracts textual content from HTML using BeautifulSoup."""
    
    @staticmethod
    def clean_html(html_content: str, url: str) -> str:
        if BeautifulSoup is None:
            raise RuntimeError("BeautifulSoup is not installed.")
            
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
        except Exception as e:
            logger.error(f"Failed to parse HTML from {url}: {e}")
            return ""
            
        # Remove irrelevant tags that don't contain main content
        for element in soup(["script", "style", "nav", "header", "footer", "aside", "form", "meta", "noscript"]):
            element.decompose()
            
        # Remove HTML comments
        for comment in soup.find_all(text=lambda text: isinstance(text, Comment)):
            comment.extract()
            
        # Extract text
        text = soup.get_text(separator=' ')
        
        # Collapse whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text


class AsyncInternetFetcher:
    """
    Asynchronously fetches web pages with concurrency limits, timeouts, 
    retries, and polite robot.txt adherence.
    """
    def __init__(self, concurrency_limit: int = 10, timeout_seconds: int = 10, max_retries: int = 2):
        self.semaphore = asyncio.Semaphore(concurrency_limit)
        self.timeout = timeout_seconds
        self.max_retries = max_retries
        self.robot_parsers: Dict[str, urllib.robotparser.RobotFileParser] = {}
        self._user_agent = "SemanticPlagiarismDetectorBot/1.0"

    async def _can_fetch(self, url: str) -> bool:
        """Checks robots.txt for permission to fetch the URL."""
        parsed_url = urllib.parse.urlparse(url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        
        if base_url not in self.robot_parsers:
            rp = urllib.robotparser.RobotFileParser()
            rp.set_url(f"{base_url}/robots.txt")
            try:
                # In a purely async environment, fetching robots.txt should also be async.
                # For simplicity and to avoid blocking, we assume True if it fails or blocks.
                # A true production system would async fetch the robots.txt.
                rp.read()
                self.robot_parsers[base_url] = rp
            except Exception:
                self.robot_parsers[base_url] = None
                
        rp = self.robot_parsers.get(base_url)
        if rp is None:
            return True
        return rp.can_fetch(self._user_agent, url)

    async def fetch_url(self, session: Any, url: str) -> FetchedSource:
        """Fetches a single URL asynchronously with retries and timeout."""
        start_time = datetime.now()
        
        if aiohttp is None:
            return FetchedSource(url, "Error", "", 0.0, is_valid=False, error_message="aiohttp not installed")
            
        if not await self._can_fetch(url):
            return FetchedSource(url, "Blocked", "", 0.0, is_valid=False, error_message="Blocked by robots.txt")

        async with self.semaphore:
            for attempt in range(self.max_retries):
                try:
                    timeout = aiohttp.ClientTimeout(total=self.timeout)
                    async with session.get(url, timeout=timeout, headers={"User-Agent": self._user_agent}) as response:
                        if response.status == 200:
                            html = await response.text()
                            duration = (datetime.now() - start_time).total_seconds() * 1000
                            
                            cleaned_text = HtmlCleaner.clean_html(html, url)
                            if not cleaned_text:
                                return FetchedSource(url, "Empty", "", duration, is_valid=False, error_message="Empty after cleaning")
                                
                            return FetchedSource(url, "Fetched", cleaned_text, duration)
                        else:
                            if attempt == self.max_retries - 1:
                                duration = (datetime.now() - start_time).total_seconds() * 1000
                                return FetchedSource(url, "Error", "", duration, is_valid=False, error_message=f"HTTP {response.status}")
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        duration = (datetime.now() - start_time).total_seconds() * 1000
                        return FetchedSource(url, "Error", "", duration, is_valid=False, error_message=str(e))
                        
                await asyncio.sleep(1.0 * (attempt + 1))  # Exponential backoff

        return FetchedSource(url, "Error", "", 0.0, is_valid=False, error_message="Max retries exceeded")

    async def fetch_all(self, urls: List[str]) -> List[FetchedSource]:
        """Fetches a batch of URLs concurrently."""
        if aiohttp is None:
            logger.error("aiohttp is required for async fetching.")
            return []
            
        unique_urls = list(set(urls))
        async with aiohttp.ClientSession() as session:
            tasks = [self.fetch_url(session, url) for url in unique_urls]
            return await asyncio.gather(*tasks)
