"""Pytest configuration for visualization tests."""

import sys
from unittest.mock import MagicMock

# Stub out broken/heavy modules so tests can collect without errors.
MOCK_MODULES = [
    "src.db",
    "src.db.auth",
    "src.db.corpus_db",
    "src.core.document_parser",
    "src.core.embedding_model",
    "src.core.faiss_index",
    "src.core.translator",
    "src.core.webhook",
    "striprtf",
    "striprtf.striprtf",
    "pdfplumber",
    "defusedxml",
    "defusedxml.lxml",
]

for mod in MOCK_MODULES:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()
