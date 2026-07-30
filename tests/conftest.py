"""
conftest.py
-----------
Global pytest fixtures and path configuration for the semantic plagiarism
detector test suite.

Path Bootstrap
~~~~~~~~~~~~~~
Inserts the repository root into sys.path so that `src`, `app`, and `utils`
packages are importable when running `pytest` directly from the project root.

This acts as a robust fallback guarantee alongside the `pythonpath = .`
directive in pytest.ini, ensuring compatibility with older pytest versions
(< 7.0) that do not support the pythonpath ini option.

Sentence Transformers Stub
~~~~~~~~~~~~~~~~~~~~~~~~~~
Stubs out sentence_transformers so tests can run without a fully compatible
TensorFlow / Keras installation. The embedding_model tests mock _get_model()
directly, so no real model is loaded.
"""

import os
import pathlib
import shutil
import sys
import types
from unittest.mock import MagicMock

import numpy as np

# ── Redis Test Database Isolation ─────────────────────────────────────────
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

import pytest

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
    class Tensor:
        pass
    torch_stub.Tensor = Tensor  # type: ignore
    sys.modules["torch"] = torch_stub


for mod_name in [
    "lxml", "defusedxml", "defusedxml.lxml", "fitz", "docx", "redis", "bs4", "faker", "argon2", "argon2.exceptions",
    "pdfplumber", "langdetect", "striprtf", "striprtf.striprtf", "src.core.translator",
    "src.core.webhook",
    "pypdf", "PyPDF2", "reportlab", "reportlab.pdfgen", "reportlab.lib", "reportlab.platypus", 
    "reportlab.lib.colors", "reportlab.lib.enums", "reportlab.lib.styles", "reportlab.lib.units", 
    "reportlab.lib.pagesizes", "reportlab.lib.utils", 
    "matplotlib", "matplotlib.patches", "matplotlib.pyplot", "matplotlib.figure", "matplotlib.ticker",
    "faiss", "torch", "psutil", "pytesseract", "sklearn", "sklearn.metrics", "sklearn.metrics.pairwise", "requests",
]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()



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

@pytest.fixture
def mock_db(tmp_path):
    """
    Provides an isolated, empty, and writable SQLite database schema for tests.
    Patches the global database paths in src.db modules to use temporary files.
    Ensures safe teardown and no interference with production databases.

    Corpus and auth use *separate* files because they each rely on PRAGMA
    user_version for migration tracking and would collide on the same file.
    """
    corpus_db_file = tmp_path / "test_corpus.db"
    auth_db_file = tmp_path / "test_users.db"

    # We patch the database path at the module level for all db modules
    import unittest.mock
    
    with unittest.mock.patch("src.db.corpus_db._DB_PATH", str(corpus_db_file)), \
         unittest.mock.patch("src.db.incidents.DEFAULT_DB_PATH", str(corpus_db_file)), \
         unittest.mock.patch("src.db.auth._DB_PATH", str(auth_db_file)):
        
        try:
            from src.db.corpus_db import init_corpus_db
            from src.db.incidents import init_incident_db
            from src.db.auth import init_db
            init_corpus_db()
            init_incident_db()
            init_db()
        except Exception:
            import traceback
            traceback.print_exc()
            
        yield str(corpus_db_file)
