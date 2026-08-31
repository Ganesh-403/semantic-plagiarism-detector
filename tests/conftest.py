"""
conftest.py
-----------
Global pytest fixtures and path configuration for the semantic plagiarism
detector test suite.

Path Bootstrap
~~~~~~~~~~~~~~
Inserts the repository root into sys.path so that `src`, `app`, and `utils`
packages are importable when running `pytest` directly from the project root.

Sentence Transformers Stub
~~~~~~~~~~~~~~~~~~~~~~~~~~
Stubs out sentence_transformers so tests can run without a fully compatible
TensorFlow / Keras installation. The embedding_model tests mock _get_model()
directly, so no real model is loaded.

Recent Additions (Issue #566):
- Added `sample_document_files` parameterized fixture supplying valid synthetic
  file buffers for PDF, DOCX, and TXT formats for comprehensive parser testing.
"""

import importlib
import io
import os
import pathlib
import shutil
import sys
import types
import zipfile
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

# ── Redis Test Database Isolation ─────────────────────────────────────────────
# Use a separate Redis database (1 instead of 0) during tests so that running
# the test suite does not flush the active development session cache.
os.environ.setdefault("REDIS_DB", "1")

# ── Headless Renderer Configuration (Issue #504) ──────────────────────────────
# Force Matplotlib to use the non-GUI Agg backend on headless CI workers
os.environ.setdefault("MPLBACKEND", "Agg")
try:
    import matplotlib

    matplotlib.use("Agg")
except ImportError:
    pass

# Patch torch.__spec__ for Python 3.13 + PyTorch compatibility
try:
    import torch

    if getattr(torch, "__spec__", None) is None:
        import importlib.util

        torch.__spec__ = importlib.util.spec_from_loader("torch", loader=None)
except ImportError:
    pass

# ── Repository Root Path Bootstrap ────────────────────────────────────────────
_REPO_ROOT = pathlib.Path(__file__).parent.parent.resolve()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ── Sentence Transformers Stub ────────────────────────────────────────────────
if "sentence_transformers" not in sys.modules:
    stub = types.ModuleType("sentence_transformers")
    stub.SentenceTransformer = MagicMock  # type: ignore[attr-defined]
    sys.modules["sentence_transformers"] = stub

if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")
    torch_stub.__path__ = []

    Tensor = type("Tensor", (), {"__module__": "torch"})
    torch_stub.Tensor = Tensor  # type: ignore
    torch_quant_stub = types.ModuleType("torch.quantization")
    torch_stub.quantization = torch_quant_stub
    sys.modules["torch"] = torch_stub
    sys.modules["torch.quantization"] = torch_quant_stub

if "faiss" not in sys.modules:
    try:
        import faiss  # noqa: F401
    except ImportError:
        faiss_stub = types.ModuleType("faiss")

        class _MockIndex:
            def __init__(self, d, *args, **kwargs):
                self.d = d
                self.vectors = []

            def add(self, x):
                if isinstance(x, np.ndarray):
                    self.vectors.append(x)

            def search(self, q, k):
                if not self.vectors:
                    return np.zeros((q.shape[0], k), dtype=np.float32), np.full(
                        (q.shape[0], k), -1, dtype=np.int64
                    )
                data = np.vstack(self.vectors)
                scores = np.dot(q, data.T)
                n_samples = data.shape[0]
                actual_k = min(k, n_samples)
                sorted_indices = np.argsort(-scores, axis=1)[:, :actual_k]
                sorted_distances = np.take_along_axis(scores, sorted_indices, axis=1)
                if actual_k < k:
                    pad_width = k - actual_k
                    sorted_indices = np.pad(
                        sorted_indices, ((0, 0), (0, pad_width)), constant_values=-1
                    )
                    sorted_distances = np.pad(
                        sorted_distances,
                        ((0, 0), (0, pad_width)),
                        constant_values=-np.inf,
                    )
                return sorted_distances.astype(np.float32), sorted_indices.astype(
                    np.int64
                )

        faiss_stub.IndexFlatIP = _MockIndex
        faiss_stub.IndexHNSWFlat = _MockIndex
        faiss_stub.METRIC_INNER_PRODUCT = 0
        sys.modules["faiss"] = faiss_stub



for mod_name in [
    "fitz",
    "redis",
    "bs4",
    "faker",
    "argon2",
    "argon2.exceptions",
    "pdfplumber",
    "langdetect",
    "striprtf",
    "striprtf.striprtf",
    "src.core.translator",
    "pypdf",
    "reportlab",
    "reportlab.pdfgen",
    "reportlab.lib",
    "reportlab.platypus",
    "reportlab.lib.colors",
    "reportlab.lib.enums",
    "reportlab.lib.styles",
    "reportlab.lib.units",
    "reportlab.lib.pagesizes",
    "reportlab.lib.utils",
    "matplotlib",
    "matplotlib.patches",
    "matplotlib.pyplot",
    "matplotlib.figure",
    "matplotlib.ticker",
    "networkx",
    "faiss",
    "torch",
    "psutil",
    "pytesseract",
    "sklearn",
    "sklearn.metrics",
    "sklearn.metrics.pairwise",
    "sklearn.feature_extraction",
    "sklearn.feature_extraction.text",
    "requests",
    "streamlit",
    "streamlit.components",
    "streamlit.components.v1",
    "transformers",
]:
    if mod_name not in sys.modules:
        try:
            importlib.import_module(mod_name)
        except Exception:
            sys.modules[mod_name] = MagicMock()

if "slowapi" not in sys.modules:
    try:
        import slowapi  # noqa: F401
    except ImportError:
        slowapi_stub = types.ModuleType("slowapi")
        slowapi_errors_stub = types.ModuleType("slowapi.errors")

        class RateLimitExceeded(Exception):
            def __init__(self, detail: str = ""):
                self.detail = detail
                super().__init__(detail)

        slowapi_errors_stub.RateLimitExceeded = RateLimitExceeded
        slowapi_middleware_stub = types.ModuleType("slowapi.middleware")

        class SlowAPIMiddleware:
            def __init__(self, app, *args, **kwargs):
                self.app = app

            async def __call__(self, scope, receive, send):
                await self.app(scope, receive, send)

        slowapi_middleware_stub.SlowAPIMiddleware = SlowAPIMiddleware
        slowapi_util_stub = types.ModuleType("slowapi.util")
        slowapi_util_stub.get_remote_address = lambda request: "127.0.0.1"

        class Limiter:
            def __init__(self, *args, **kwargs):
                pass

            def limit(self, *args, **kwargs):
                def decorator(func):
                    return func

                return decorator

            def _inject_headers(self, response, *args, **kwargs):
                return response

        slowapi_stub.Limiter = Limiter
        sys.modules["slowapi"] = slowapi_stub
        sys.modules["slowapi.errors"] = slowapi_errors_stub
        sys.modules["slowapi.middleware"] = slowapi_middleware_stub
        sys.modules["slowapi.util"] = slowapi_util_stub


# ── Tesseract OCR Availability ────────────────────────────────────────────────
TESSERACT_AVAILABLE = shutil.which("tesseract") is not None


@pytest.fixture
def sqlite_database_path(tmp_path):
    """Return an isolated SQLite path for migration/database tests."""
    return tmp_path / "test.db"


# ── Consolidated Application Fixtures (Issue #372) ───────────────────────────
@pytest.fixture(autouse=True)
def clean_test_env():
    """
    Globally auto-used fixture that cleans up the FAISS index and SQLite DB
    before and after every test, preventing state leakage across test cases.
    """
    try:
        from src.db.corpus_db import clear_all_data

        clear_all_data()
    except Exception:
        try:
            from src.db.corpus_db import close_connections

            close_connections()
        except Exception:
            pass

    index_path = os.path.join(str(_REPO_ROOT), "corpus.index")
    db_path = os.path.join(str(_REPO_ROOT), "corpus.db")
    users_db_path = os.path.join(str(_REPO_ROOT), "users.db")

    for path in [index_path, db_path, users_db_path]:
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass

    yield

    try:
        from src.db.corpus_db import clear_all_data

        clear_all_data()
    except Exception:
        try:
            from src.db.corpus_db import close_connections

            close_connections()
        except Exception:
            pass

    for path in [index_path, db_path, users_db_path]:
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass


@pytest.fixture
def dummy_embeddings():
    """
    Consolidated dummy embeddings for similarity and core tests.
    Returns 384-dimensional fake embeddings for 3 documents.
    """
    emb_a = np.array([[1.0, 0.0, 0.0], [0.8, 0.6, 0.0]])
    emb_b = np.array([[0.9, 0.1, 0.0], [0.8, 0.5, 0.0]])
    emb_c = np.array([[0.0, 0.0, 1.0]])
    return {"doc_A": emb_a, "doc_B": emb_b, "doc_C": emb_c}


class MockDataFactory:
    """
    Generalized factory pattern for generating test mocks.
    Consolidates multiple disparate mocking functions.
    """

    @staticmethod
    def embed_chunks(chunks, batch_size=64):
        """Standardized fast embedding mock for streamlit app tests."""
        if not chunks:
            return np.array([])
        val = 1.0 / (384**0.5)
        return np.full((len(chunks), 384), val, dtype="float32")


@pytest.fixture
def mock_factory():
    """Returns the consolidated MockDataFactory for tests."""
    return MockDataFactory()


@pytest.fixture
def mock_embed_chunks():
    """Provides a direct reference to the embed_chunks factory method."""
    return MockDataFactory.embed_chunks


import time
import logging
import abc
import typing
from typing import List, Optional, Set
from pathlib import Path

# -----------------------------------------------------------------------------
# Enterprise Database Lifecycle Management
# -----------------------------------------------------------------------------
class AbstractTeardownStrategy(abc.ABC):
    """Abstract base class for all file and connection teardown strategies."""
    @abc.abstractmethod
    def execute_teardown(self, target_path: Path) -> bool:
        pass

class SQLiteConnectionTeardownStrategy(AbstractTeardownStrategy):
    """Safely terminates dangling SQLite connections to prevent Win32 file lock exceptions."""
    def execute_teardown(self, target_path: Path) -> bool:
        try:
            # Force close connections from known singleton caches
            from src.db.corpus_db import close_connections
            close_connections(all_threads=True)
            return True
        except ImportError:
            return False
        except Exception as e:
            logging.error(f"Failed to close SQLite connections: {e}")
            return False

class ExponentialBackoffFileUnlinkStrategy(AbstractTeardownStrategy):
    """Attempts to unlink files with exponential backoff to handle transient OS locks."""
    def __init__(self, max_retries: int = 5, initial_backoff: float = 0.05):
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff

    def execute_teardown(self, target_path: Path) -> bool:
        if not target_path.exists():
            return True
            
        for attempt in range(self.max_retries):
            try:
                target_path.unlink()
                return True
            except OSError as e:
                if attempt == self.max_retries - 1:
                    logging.error(f"Failed to unlink {target_path} after {self.max_retries} attempts: {e}")
                    return False
                time.sleep(self.initial_backoff * (2 ** attempt))
        return False

class EnterpriseFixtureTeardownManager:
    """Orchestrates complex teardown logic across multi-file database artifacts."""
    def __init__(self) -> None:
        self.strategies: List[AbstractTeardownStrategy] = [
            SQLiteConnectionTeardownStrategy(),
            ExponentialBackoffFileUnlinkStrategy()
        ]
        self.tracked_files: Set[Path] = set()

    def track_database(self, db_path: Path) -> None:
        """Tracks the primary database and its associated WAL/SHM artifacts."""
        self.tracked_files.add(db_path)
        self.tracked_files.add(db_path.with_suffix(db_path.suffix + "-wal"))
        self.tracked_files.add(db_path.with_suffix(db_path.suffix + "-shm"))
        self.tracked_files.add(db_path.with_suffix(db_path.suffix + "-journal"))

    def execute_all(self) -> None:
        """Executes all teardown strategies across all tracked files."""
        # Step 1: Close connections first
        connection_strategy = self.strategies[0]
        connection_strategy.execute_teardown(Path("."))
        
        # Step 2: Unlink all files
        unlink_strategy = self.strategies[1]
        for file_path in self.tracked_files:
            unlink_strategy.execute_teardown(file_path)

@pytest.fixture
def mock_db(tmp_path):
    """
    Provides an isolated, empty, and writable SQLite database schema for tests.
    Patches the global database paths in src.db modules to use temporary files.
    Includes highly-engineered, fail-safe teardown logic to prevent test pollution.
    """
    corpus_db_file = tmp_path / "test_corpus.db"
    auth_db_file = tmp_path / "test_users.db"
    
    manager = EnterpriseFixtureTeardownManager()
    manager.track_database(corpus_db_file)
    manager.track_database(auth_db_file)

    import unittest.mock

    with (
        unittest.mock.patch("src.db.corpus_db._DB_PATH", str(corpus_db_file)),
        unittest.mock.patch("src.db.incidents.DEFAULT_DB_PATH", str(corpus_db_file)),
        unittest.mock.patch("src.db.auth._DB_PATH", str(auth_db_file)),
    ):
        try:
            from src.db.auth import init_db
            from src.db.corpus_db import init_corpus_db
            from src.db.incidents import init_incident_db

            init_corpus_db()
            init_incident_db()
            init_db()
        except Exception:
            import traceback
            traceback.print_exc()

        yield str(corpus_db_file)
        
        # Acceptance Criteria Teardown execution
        manager.execute_all()


# ── Multi-Format Sample Files Fixture (Issue #566) ───────────────────────────
@pytest.fixture(params=["pdf", "docx", "txt"])
def sample_document_files(request):
    """
    Parameterized fixture supplying valid synthetic file buffers for PDF, DOCX, and TXT.

    This fixture is essential for testing the document parsing pipeline
    (`src.core.document_parser`) without relying on external, static test files.
    It generates minimal, structurally valid file formats in memory.

    Yields:
        tuple: (io.BytesIO buffer, str filename)
    """
    file_type = request.param

    if file_type == "txt":
        # Standard plain text file
        content = b"This is a sample text document for testing purposes.\nIt contains multiple lines to verify line-by-line parsing.\n"
        filename = "sample_test.txt"
        yield io.BytesIO(content), filename

    elif file_type == "pdf":
        # Minimal valid PDF 1.4 structure
        # Contains Catalog, Pages, and a single Page object to satisfy basic parsers
        pdf_content = (
            b"%PDF-1.4\n"
            b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
            b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
            b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n"
            b"xref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n"
            b"trailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n190\n%%EOF\n"
        )
        filename = "sample_test.pdf"
        yield io.BytesIO(pdf_content), filename

    elif file_type == "docx":
        # Minimal valid DOCX structure (ZIP archive with required Office Open XML files)
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            # 1. Content Types
            zf.writestr(
                "[Content_Types].xml",
                b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                b'<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                b'<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                b'<Default Extension="xml" ContentType="application/xml"/>'
                b'<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
                b"</Types>",
            )
            # 2. Main Document
            zf.writestr(
                "word/document.xml",
                b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                b'<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                b"<w:body>"
                b"<w:p><w:r><w:t>Sample DOCX content for testing the parsing pipeline.</w:t></w:r></w:p>"
                b"</w:body>"
                b"</w:document>",
            )
            # 3. Relationships
            zf.writestr(
                "_rels/.rels",
                b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                b'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                b'<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
                b"</Relationships>",
            )
        zip_buffer.seek(0)
        filename = "sample_test.docx"
        yield zip_buffer, filename


import sqlite3
from pathlib import Path


@pytest.fixture(autouse=True)
def clear_streamlit_singletons():
    try:
        from streamlit.delta_generator_singletons import _dg_singleton

        _dg_singleton._instance = None
    except ImportError:
        pass


@pytest.fixture(autouse=True)
def _cleanup_corpus_db_connections():
    yield
    try:
        from src.db.corpus_db import close_connections

        close_connections(all_threads=True)
    except ImportError:
        pass


@pytest.fixture
def db_connection(tmp_path: Path) -> sqlite3.Connection:
    """Provide a clean, initialized SQLite database connection for testing.

    This fixture creates a temporary SQLite database in the pytest tmp_path,
    initializes the required schema (incidents, documents, etc.), yields the
    active connection for the test to use, and automatically closes the
    connection during teardown.

    This eliminates the need for manual sqlite3.connect() and conn.close()
    calls in every test function (Issue #2725).

    Yields:
        sqlite3.Connection: An active, initialized database connection.
    """
    db_path = tmp_path / "test_plagiarism.db"

    # Create connection with row factory for dictionary-like access
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Initialize schema (simplified for test environment)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT UNIQUE NOT NULL,
            file_hash TEXT UNIQUE,
            upload_date TEXT NOT NULL,
            class_section TEXT,
            student_name TEXT,
            is_deleted INTEGER DEFAULT 0
        );
        
        CREATE TABLE IF NOT EXISTS plagiarism_incidents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            incident_id TEXT UNIQUE NOT NULL,
            document_a TEXT NOT NULL,
            document_b TEXT NOT NULL,
            similarity REAL NOT NULL,
            severity TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            threshold_at_time_of_flag REAL,
            review_status TEXT DEFAULT 'Pending'
        );
        
        CREATE INDEX IF NOT EXISTS idx_incidents_docs 
        ON plagiarism_incidents(document_a, document_b);
    """)
    conn.commit()

    # Yield the connection to the test
    yield conn

    # Teardown: Close the connection
    conn.close()


@pytest.fixture
def populated_db_connection(db_connection: sqlite3.Connection) -> sqlite3.Connection:
    """Provide a database connection pre-populated with sample incident data.

    Builds on the base db_connection fixture by inserting 50 sample
    plagiarism incidents with varying severities and similarities.
    """
    sample_incidents = []
    for i in range(50):
        sim = 0.50 + (i * 0.01)
        severity = "High" if sim >= 0.80 else ("Medium" if sim >= 0.60 else "Low")
        sample_incidents.append(
            (
                f"INC-{i:04d}",
                f"student_{i}_a.pdf",
                f"student_{i}_b.pdf",
                sim,
                severity,
                f"2024-01-{(i % 28) + 1:02d}T10:00:00",
                0.59,
                "Pending",
            )
        )

    db_connection.executemany(
        """
        INSERT INTO plagiarism_incidents 
        (incident_id, document_a, document_b, similarity, severity, timestamp, threshold_at_time_of_flag, review_status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        sample_incidents,
    )
    db_connection.commit()
    return db_connection


@pytest.fixture
def mock_fast_tokenizer(monkeypatch):
    """
    A lightweight deterministic mock tokenizer that produces fixed-length token arrays.
    Prevents unit tests from downloading/loading massive PyTorch/HuggingFace models.
    """
    from unittest.mock import MagicMock
    import torch

    class MockFastTokenizer(MagicMock):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.model_max_length = 512
            self.pad_token_id = 0
            self.eos_token_id = 2
            self.bos_token_id = 1
            
        def __call__(self, text, *args, **kwargs):
            if isinstance(text, str):
                texts = [text]
            else:
                texts = text
                
            batch_size = len(texts)
            # Dummy fixed-length array
            seq_len = 16 
            
            input_ids = torch.ones((batch_size, seq_len), dtype=torch.long)
            attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
            
            # Make it deterministic based on input length
            for i, t in enumerate(texts):
                length = min(len(t) // 4 + 1, seq_len)
                input_ids[i, :length] = torch.arange(1, length + 1)
                attention_mask[i, length:] = 0
                
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }

    tokenizer = MockFastTokenizer()
    
    # Mock AutoModelForSequenceClassification to avoid loading it
    mock_model = MagicMock()
    mock_model.config.max_position_embeddings = 512
    
    # Mock loss to return a tensor with a valid value so perplexity does not crash
    mock_outputs = MagicMock()
    mock_outputs.loss.item.return_value = 1.0
    type(mock_outputs.loss).__float__ = MagicMock(return_value=1.0)
    mock_model.return_value = mock_outputs
    
    try:
        import transformers
        monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer)
        monkeypatch.setattr(transformers.AutoModelForSequenceClassification, "from_pretrained", lambda *args, **kwargs: mock_model)
    except ImportError:
        pass
        
    try:
        import sentence_transformers
        monkeypatch.setattr(sentence_transformers, "SentenceTransformer", lambda *args, **kwargs: MagicMock())
    except ImportError:
        pass
        
    return tokenizer
