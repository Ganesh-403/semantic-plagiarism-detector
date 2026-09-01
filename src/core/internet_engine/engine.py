"""
Orchestrator for the Internet Plagiarism Module (Issue #4260).
Integrates keyword extraction, search, fetching, and transient indexing.
"""

import asyncio
import logging
from datetime import datetime
from typing import List, Dict

from src.core.internet_engine.models import (
    SearchQuery, InternetPlagiarismReport, TransientIndexConfig
)
from src.core.internet_engine.scraper import (
    QueryGenerator, EnvironmentConfiguredSearchProvider, AsyncInternetFetcher
)
from src.core.internet_engine.transient_index import TransientInternetIndex

logger = logging.getLogger(__name__)

class InternetPlagiarismEngine:
    """
    High-level engine that manages the end-to-end process of detecting 
    plagiarism from internet sources.
    """
    def __init__(self, config: TransientIndexConfig = None):
        self.config = config or TransientIndexConfig()
        self.query_generator = QueryGenerator()
        self.search_provider = EnvironmentConfiguredSearchProvider()
        self.fetcher = AsyncInternetFetcher(concurrency_limit=5, timeout_seconds=10)

    async def scan_document_async(self, doc_id: str, text: str) -> InternetPlagiarismReport:
        """
        Asynchronously executes the full internet scraping and indexing pipeline.
        """
        start_time = datetime.now()
        report = InternetPlagiarismReport(submitted_doc_id=doc_id, total_queries_run=0, total_urls_fetched=0)
        
        if not text.strip():
            logger.warning("Empty text submitted to Internet Engine.")
            return report

        # 1. Generate Queries
        queries = self.query_generator.generate_queries(doc_id, text)
        if not queries:
            return report
            
        report.total_queries_run = len(queries)
        
        # 2. Execute Searches
        all_urls = set()
        for query in queries:
            try:
                results = await self.search_provider.search(query, limit=3)
                for res in results:
                    all_urls.add(res.url)
            except Exception as e:
                logger.error(f"Search provider failed for query '{query.query_text}': {e}")
                
        if not all_urls:
            logger.info("No URLs found from search provider.")
            return report

        # 3. Fetch Internet Sources
        urls_list = list(all_urls)
        fetched_sources = await self.fetcher.fetch_all(urls_list)
        
        valid_sources = []
        for src in fetched_sources:
            if src.is_valid:
                valid_sources.append(src)
            else:
                report.failed_urls[src.url] = src.error_message or "Unknown error"
                
        report.total_urls_fetched = len(valid_sources)
        
        # 4. Build Transient Index and Compare
        if valid_sources:
            index = TransientInternetIndex(self.config)
            index.build_index(valid_sources)
            
            matches = index.search(doc_id, text)
            report.matches = matches

        # 5. Finalize Report
        report.execution_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        return report

    def scan_document_sync(self, doc_id: str, text: str) -> InternetPlagiarismReport:
        """
        Synchronous wrapper for `scan_document_async`. 
        Safely runs the asyncio event loop.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We are already in an event loop (e.g. running inside Streamlit/Jupyter)
                # Nest_asyncio would be needed here, or return a coroutine. 
                # For this implementation we'll assume standard synchronous calling context
                # or create a new loop.
                raise RuntimeError("Cannot run sync wrapper inside an active async event loop.")
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        return loop.run_until_complete(self.scan_document_async(doc_id, text))
