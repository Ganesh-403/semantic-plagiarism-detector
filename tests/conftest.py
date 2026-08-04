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

import io
import os
import pathlib
import shutil
import sys
import types
import zipfile
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


import importlib.util

for mod_name in [
    "fitz", "redis", "bs4", "faker", "argon2", "argon2.exceptions",
    "pdfplumber", "langdetect", "striprtf", "striprtf.striprtf", "src.core.translator",
    "src.core.webhook",
    "pypdf", "reportlab", "reportlab.pdfgen", "reportlab.lib", "reportlab.platypus",
    "reportlab.lib.colors", "reportlab.lib.enums", "reportlab.lib.styles", "reportlab.lib.units",
    "reportlab.lib.pagesizes", "reportlab.lib.utils",
    "matplotlib", "matplotlib.patches", "matplotlib.pyplot", "matplotlib.figure", "matplotlib.ticker",
    "networkx",
    "faiss", "torch", "psutil", "pytesseract",
    "sklearn", "sklearn.metrics", "sklearn.metrics.pairwise",
    "sklearn.feature_extraction", "sklearn.feature_extraction.text",
    "requests",
    "streamlit", "streamlit.components", "streamlit.components.v1",
    "transformers",
]:
    if mod_name not in sys.modules:
        try:
            spec = importlib.util.find_spec(mod_name)
            if spec is None:
                top_pkg = mod_name.split(".")[0]
                if importlib.util.find_spec(top_pkg) is None:
                    sys.modules[mod_name] = MagicMock()
        except Exception:
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
    """
    corpus_db_file = tmp_path / "test_corpus.db"
    auth_db_file = tmp_path / "test_users.db"

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
                b'</Types>'
            )
            # 2. Main Document
            zf.writestr(
                "word/document.xml",
                b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                b'<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                b'<w:body>'
                b'<w:p><w:r><w:t>Sample DOCX content for testing the parsing pipeline.</w:t></w:r></w:p>'
                b'</w:body>'
                b'</w:document>'
            )
            # 3. Relationships
            zf.writestr(
                "_rels/.rels",
                b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                b'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                b'<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
                b'</Relationships>'
            )
        zip_buffer.seek(0)
        filename = "sample_test.docx"
        yield zip_buffer, filename
