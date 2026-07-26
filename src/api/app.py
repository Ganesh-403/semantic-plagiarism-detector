"""src/api/app.py - FastAPI REST API for LMS integration."""

import logging
import os
from typing import Dict

import numpy as np
from fastapi import (Depends, FastAPI, File, HTTPException, Query, UploadFile,
                     status)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from sklearn.metrics.pairwise import cosine_similarity

from src.core.document_parser import extract_text
from src.core.embedding_model import embed_chunks, get_document_embedding
from src.core.similarity import (PLAGIARISM_THRESHOLD, chunk_max_similarity,
                                 find_most_similar_chunks)
from src.core.text_chunking import chunk_document
from src.db.auth import get_user_role
from src.db.corpus_db import _connect, clear_all_data, init_corpus_db
from src.utils.redis_cache import get_cache

# ── API Initialization ────────────────────────────────────────────────────────

app = FastAPI(
    title="Semantic Plagiarism Detector API",
    description="REST API for programmatically checking documents for semantic plagiarism.",
    version="1.0.0",
)

# Enable CORS for external LMS frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Bearer Token Authentication ────────────────────────────────────────────────

security = HTTPBearer()


def get_expected_bearer_token() -> str:
    """Retrieve the API Bearer Token from environment variable or default fallback."""
    return os.getenv("API_BEARER_TOKEN", "dev-bearer-token")


def verify_bearer_token(
    credentials=Depends(security),
) -> str:
    """Validate incoming Bearer token against configured secret."""
    expected_token = get_expected_bearer_token()
    if not credentials or credentials.credentials != expected_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing authentication token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials


# ── Database Helpers ───────────────────────────────────────────────────────────


def get_corpus_documents_with_embeddings() -> Dict[str, Dict]:
    """Load all stored corpus documents, text chunks, and chunk embeddings from SQLite."""
    init_corpus_db()
    corpus: Dict[str, Dict] = {}

    with _connect() as conn:
        rows = conn.execute(
            "SELECT filename, chunk_index, chunk_text, embedding FROM chunks ORDER BY filename, chunk_index"
        ).fetchall()

    for filename, _chunk_index, chunk_text, embedding_blob in rows:
        if filename not in corpus:
            corpus[filename] = {"chunks": [], "embeddings": []}

        vec = np.frombuffer(embedding_blob, dtype=np.float32)
        corpus[filename]["chunks"].append(chunk_text)
        corpus[filename]["embeddings"].append(vec)

    # Convert list of vectors into stacked 2D numpy arrays
    for filename in corpus:
        vecs = corpus[filename]["embeddings"]
        corpus[filename]["embeddings"] = (
            np.vstack(vecs) if vecs else np.empty((0, 384), dtype=np.float32)
        )

    return corpus


# ── API Endpoints ──────────────────────────────────────────────────────────────


@app.get("/health", tags=["Health"])
def health_check():
    """Healthcheck endpoint for readiness and liveness probes."""
    return {
        "status": "healthy",
        "service": "Semantic Plagiarism Detector API",
        "version": "1.0.0",
    }


_HEALTHZ_DB_PATHS = (
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "corpus.db")),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "users.db")),
)


@app.get("/healthz", tags=["Health"])
def healthz():
    """Lightweight /healthz endpoint for DevOps monitoring and load balancer probes.

    Returns 200 OK with the combined SQLite database size so operators can
    monitor storage growth without rendering the Streamlit UI.
    """
    total_bytes = 0
    for path in _HEALTHZ_DB_PATHS:
        try:
            total_bytes += os.path.getsize(path) if os.path.exists(path) else 0
        except OSError:
            pass

    return {
        "status": "ok",
        "db_size_bytes": total_bytes,
        "db_size_mb": round(total_bytes / (1024 * 1024), 2),
    }


@app.post("/api/v1/scan", tags=["Plagiarism Detection"])
async def scan_document(
    file: UploadFile = File(
        ..., description="Document file to scan (.pdf, .docx, .txt)"
    ),
    threshold: float = Query(
        default=PLAGIARISM_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for flagging plagiarism (default: 0.59)",
    ),
    top_k: int = Query(
        default=3,
        ge=1,
        le=10,
        description="Number of top matching paragraph pairs to include per matched document",
    ),
    _token: str = Depends(verify_bearer_token),
):
    """Scan an uploaded document against the indexed corpus database for plagiarism."""
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename must be provided.",
        )

    filename = file.filename
    file_bytes = await file.read()

    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    # Extract text from uploaded document
    extracted_text = extract_text(file_bytes, filename)
    if not extracted_text.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Failed to extract readable text from the uploaded file.",
        )

    words = extracted_text.split()
    word_count = len(words)

    # Split document into paragraph-level chunks
    chunks = chunk_document(extracted_text)
    if not chunks:
        # Fallback if text is shorter than MIN_CHUNK_WORDS
        chunks = [extracted_text[:1000]]

    # Generate chunk embeddings
    uploaded_embeddings = embed_chunks(chunks)

    # Compute overall single document embedding
    doc_embedding = get_document_embedding(uploaded_embeddings)

    # Query corpus from SQLite database
    corpus_docs = get_corpus_documents_with_embeddings()

    matched_documents = []
    max_overall_score = 0.0
    max_chunk_overall_score = 0.0

    for corpus_filename, corpus_data in corpus_docs.items():
        # Avoid self-comparison if the same document is in the corpus
        if corpus_filename == filename:
            continue

        c_embeddings = corpus_data["embeddings"]
        c_chunks = corpus_data["chunks"]

        if c_embeddings.size == 0:
            continue

        # Document-level mean similarity
        c_doc_embedding = get_document_embedding(c_embeddings)
        sim_doc = float(
            np.clip(
                cosine_similarity(
                    doc_embedding.reshape(1, -1), c_doc_embedding.reshape(1, -1)
                )[0, 0],
                0.0,
                1.0,
            )
        )

        # Chunk-level max similarity
        sim_chunk = chunk_max_similarity(uploaded_embeddings, c_embeddings)

        combined_score = max(sim_doc, sim_chunk)
        max_overall_score = max(max_overall_score, sim_doc)
        max_chunk_overall_score = max(max_chunk_overall_score, sim_chunk)

        if combined_score >= threshold:
            severity = "🔴 High" if combined_score >= 0.90 else "🟡 Medium"

            # Find top matching chunk pairs
            similar_chunks = find_most_similar_chunks(
                chunks_a=chunks,
                chunks_b=c_chunks,
                emb_a=uploaded_embeddings,
                emb_b=c_embeddings,
                top_k=top_k,
                threshold=threshold,
            )

            flagged_chunks = [
                {
                    "uploaded_chunk": pair[0],
                    "matched_chunk": pair[1],
                    "similarity_score": round(float(pair[2]), 4),
                }
                for pair in similar_chunks
            ]

            matched_documents.append(
                {
                    "filename": corpus_filename,
                    "document_similarity_score": round(sim_doc, 4),
                    "max_chunk_similarity_score": round(sim_chunk, 4),
                    "severity": severity,
                    "flagged_chunks": flagged_chunks,
                }
            )

    # Sort matches by max chunk similarity descending
    matched_documents.sort(key=lambda x: x["max_chunk_similarity_score"], reverse=True)

    is_flagged = len(matched_documents) > 0 or max_chunk_overall_score >= threshold

    return {
        "filename": filename,
        "word_count": word_count,
        "chunk_count": len(chunks),
        "plagiarism_flagged": is_flagged,
        "threshold_used": threshold,
        "overall_document_similarity": round(max_overall_score, 4),
        "max_chunk_similarity": round(max_chunk_overall_score, 4),
        "matched_documents_count": len(matched_documents),
        "matched_documents": matched_documents,
    }


# ── System Administration ──────────────────────────────────────────────────────

logger = logging.getLogger(__name__)
INDEX_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "corpus.index")
)


@app.post("/api/v1/clear", tags=["System Administration"])
async def clear_all_documents(
    username: str = Query(
        ..., description="Username of the administrator executing the operation"
    ),
    _token: str = Depends(verify_bearer_token),
):
    """
    Remove all documents, text chunks, and plagiarism incidents from the SQLite database,
    delete the FAISS index file, and clear the Redis cache. Restricted to administrators.
    """
    # 1. Verify administrator permissions
    role = get_user_role(username)
    if role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: Only administrators are authorized to clear all documents.",
        )

    try:
        # 2. Clear SQLite database (documents, chunks, incidents)
        clear_all_data()

        # 3. Clear/reset the FAISS index file on disk
        if os.path.exists(INDEX_PATH):
            try:
                os.remove(INDEX_PATH)
            except OSError as e:
                logger.error(f"Failed to remove FAISS index file: {e}")

        # 4. Invalidate Redis cache
        try:
            cache = get_cache()
            if cache.is_available():
                cache.delete("faiss:index:corpus_index")
                cache.clear_pattern("analysis:*")
        except Exception as e:
            logger.error(f"Failed to clear Redis cache: {e}")

        return {
            "status": "success",
            "message": "All documents, chunks, and plagiarism incidents have been cleared, and the FAISS index reset successfully.",
        }

    except Exception as e:
        logger.error(f"Error during bulk clearing: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while clearing the corpus: {str(e)}",
        )
