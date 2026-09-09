"""Behavioral regressions for the remaining issue backlog."""

import asyncio
import importlib
import io
import random
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest


def test_session_keys_are_unique():
    from app.session_keys import SessionKeys

    values = [v.value for v in SessionKeys.__members__.values()]
    assert len(values) == len(set(values))


def test_lexical_metrics():
    from src.core.lexical_similarity import (
        jaccard_similarity,
        overlap_coefficient,
        levenshtein_similarity,
    )

    rng = random.Random(4026)
    for _ in range(10):
        a, b = (
            " ".join(rng.choices(["alpha", "beta", "gamma", "delta"], k=12))
            for _ in range(2)
        )
        assert jaccard_similarity(a, b) == jaccard_similarity(b, a)
    assert overlap_coefficient("alpha beta", "alpha beta gamma") == 1
    assert overlap_coefficient("", "alpha") == 0
    assert levenshtein_similarity("kitten", "sitting") == pytest.approx(4 / 7)
    assert levenshtein_similarity("", "") == 1
    assert levenshtein_similarity("abc", "xyz") == 0
    assert levenshtein_similarity("same", "same") == 1


def test_empty_embeddings(monkeypatch):
    from src.core import embedding_model as module

    model = MagicMock()
    model.get_sentence_embedding_dimension.return_value = 384
    monkeypatch.setattr(module, "_get_model", lambda: model)
    assert module.embed_chunks([]).shape == (0, 384)
    model.encode.assert_not_called()


def test_low_memory_model(monkeypatch, caplog):
    from src.core import embedding_model as module

    monkeypatch.setattr(module, "_model", None)
    monkeypatch.setattr(module, "_quantized_model", None)
    monkeypatch.setattr(module, "_active_model_name", None)
    monkeypatch.setattr(module, "_ONNX_AVAILABLE", False)
    monkeypatch.setattr(module, "_repair_corrupted_model_cache", lambda *a: None)
    monkeypatch.setattr(
        module.psutil, "virtual_memory", lambda: MagicMock(available=1024**3)
    )
    monkeypatch.setattr(module, "_detect_device", lambda *a: "cpu")
    loader = MagicMock()
    loader.return_value.get_sentence_embedding_dimension.return_value = 384
    monkeypatch.setattr(module, "SentenceTransformer", loader)
    module.EmbeddingModelManager().get_model()
    assert loader.call_args.args[0] == "all-MiniLM-L6-v2"
    assert module.get_embedding_model_info() == ("all-MiniLM-L6-v2", 384)
    assert "below 1.5 GiB" in caplog.text


def test_chunk_sections_and_short_tail():
    from src.core.text_chunking import chunk_document

    text = (
        "# First\n"
        + "alpha beta gamma delta " * 4
        + "\n# Second\n"
        + "epsilon zeta eta theta " * 4
    )
    chunks = chunk_document(text, chunk_size=500, min_words=1)
    assert len(chunks) == 2
    assert chunks[1].startswith("# Second")
    assert chunks[1].section_title == "Second"
    pieces = chunk_document(
        "word " * 23,
        chunk_size=100,
        chunk_overlap=0,
        min_words=1,
        sentence_padding=False,
    )
    assert len(pieces) == 1
    assert pieces[0].count("word") == 23


def test_translation_errors_and_fidelity(monkeypatch, caplog):
    from src.core.translator import is_translation_error
    from src.core import cross_lingual

    assert is_translation_error("(Translation Error: timeout)")
    assert not is_translation_error(None)
    assert not is_translation_error("translated")
    assert (
        cross_lingual.verify_semantic_fidelity(np.array([1.0, 0]), np.array([0.0, 1]))
        == 0
    )
    assert "Low semantic fidelity" in caplog.text
    monkeypatch.setattr(cross_lingual, "MIN_DETECTION_CHARACTERS", 100)
    assert (
        cross_lingual.detect_chunk_language("el estudiante de la universidad") == "en"
    )
    monkeypatch.setattr(cross_lingual, "MIN_DETECTION_CHARACTERS", 1)
    assert (
        cross_lingual.detect_chunk_language("el estudiante de la universidad") == "es"
    )


def test_academic_stats():
    from src.utils.text_stats import compute_text_stats, count_sentences

    assert count_sentences("Smith et al. (2020) report a result.") == 1
    stats = compute_text_stats("Four small words here.")
    assert stats["avg_word_length"] == 4.5
    assert stats["avg_sentence_length"] == 4


@pytest.mark.parametrize("angle", [0, 90, 180, 270])
def test_encrypted_rotated_pdf_colors(angle):
    import fitz
    from src.utils.pdf_highlighter import highlight_pdf_matches

    with fitz.open() as doc:
        page = doc.new_page(width=300, height=600)
        page.insert_text((20, 500), "A matching phrase at the bottom.")
        original_rect = page.search_for("matching phrase")[0]
        page.set_rotation(angle)
        data = doc.tobytes(
            encryption=fitz.PDF_ENCRYPT_AES_256, user_pw="secret123", owner_pw="owner"
        )
    result = highlight_pdf_matches(
        data, [("matching phrase", 0.95)], password="secret123"  # pragma: allowlist secret -- synthetic test credential
    )
    with fitz.open(stream=result, filetype="pdf") as doc:
        if doc.is_encrypted:
            doc.authenticate("secret123")
        page = doc[0]
        annotations = list(page.annots())
        assert len(annotations) == 1
        assert annotations[0].colors["stroke"] == [1.0, 0.0, 0.0]
        vertices = annotations[0].vertices
        assert abs(vertices[0][1] - original_rect.y0) < 1


def test_version_prereleases():
    from src.utils.version_check import is_update_available

    assert not is_update_available("1.0.0", "v2.0.0b1")
    assert is_update_available("1.0.0", "v2.0.0b1", allow_prereleases=True)
    assert is_update_available("2.0.0a1", "v2.0.0b1")
    assert is_update_available("1.0.0", "v1.1.0")


@pytest.mark.asyncio
async def test_version_network_failure_backoff(monkeypatch):
    from src.utils import version_check as module

    module.clear_version_cache()
    clock = [1000.0]
    monkeypatch.setattr(module.time, "time", lambda: clock[0])
    client = AsyncMock()
    client.get.side_effect = OSError("offline")
    factory = MagicMock()
    factory.return_value.__aenter__.return_value = client
    monkeypatch.setattr(module.httpx, "AsyncClient", factory)
    try:
        await module.fetch_latest_github_version()
        clock[0] += 899
        await module.fetch_latest_github_version()
        assert client.get.call_count == 1
        clock[0] += 2
        await module.fetch_latest_github_version()
        assert client.get.call_count == 2
    finally:
        module.clear_version_cache()


def test_corpus_delete_cascades_ten_chunks(mock_db):
    from src.db.corpus_db import add_document, add_chunks, _connect

    add_document("cascade.txt", "cascade-hash")
    add_chunks(
        [
            (i, "cascade.txt", i, f"chunk {i}", np.ones(384, dtype=np.float32))
            for i in range(10)
        ]
    )
    with _connect() as conn:
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM chunks WHERE filename=?", ("cascade.txt",)
            ).fetchone()[0]
            == 10
        )
        conn.execute("DELETE FROM documents WHERE filename=?", ("cascade.txt",))
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM chunks WHERE filename=?", ("cascade.txt",)
            ).fetchone()[0]
            == 0
        )


def test_faiss_concurrent_add_search_and_save(tmp_path, monkeypatch):
    import faiss
    from src.core import faiss_index as module, app_config

    path = tmp_path / "index.faiss"
    monkeypatch.setattr(app_config, "FAISS_INDEX_PATH", path)
    index = faiss.IndexFlatIP(3)
    index.add(np.array([[1, 0, 0]], dtype=np.float32))
    index, registry = module.add_to_index(
        index,
        [module.ChunkRecord("old", 0, "old")],
        {"new": np.array([[0, 1, 0]], dtype=np.float32)},
        {"new": ["new"]},
    )
    assert faiss.read_index(str(path)).ntotal == 2
    manager = module.FaissIndexManager(index, dimension=3)

    def operation(i):
        if i % 2:
            manager.add([[1, 0, 0]])
        else:
            scores, ids = manager.search([[1, 0, 0]], 1)
            assert scores[0, 0] == pytest.approx(1)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(operation, range(20)))
    assert manager.ntotal == 12


def test_neural_clone_contract():
    from src.services.neural_code_clone_engine import NeuralCodeCloneEngine as engine
    from src.models.neural_code_clone_model import CodeAstEmbedding, CodeCloneMatch

    source = CodeAstEmbedding("a", "python", 10, 2, [1, 0])
    target = CodeAstEmbedding("b", "python", 10, 2, [1, 0])
    result = engine.analyze_code_pair("a", "b", source, target)
    assert isinstance(result, CodeCloneMatch)
    assert result.overall_clone_score == 1
    assert engine.calculate_token_cosine_similarity([0, 0], [1, 0]) == 0


def test_card_modules_import():
    for name in ["code_clone", "faiss_vector", "multimodal_ocr", "stylometric_author"]:
        importlib.import_module(f"src.components.{name}_card")


def test_auth_failure_metrics(mock_db, monkeypatch):
    from src.db.auth import add_user, verify_user
    from src.security.jwt_utils import (
        create_jwt_token,
        verify_access_token,
        JWTExpiredError,
    )
    from src.core.metrics import auth_failures_total

    add_user("metrics_user", "Valid-password-2026!", "teacher")
    before = auth_failures_total.labels(reason="invalid_password")._value.get()
    assert not verify_user("metrics_user", "wrong-password")
    assert (
        auth_failures_total.labels(reason="invalid_password")._value.get() == before + 1
    )
    monkeypatch.setenv("JWT_SECRET_KEY", "test-only-metrics-secret-key-12345")
    token = create_jwt_token(
        {"sub": "metrics_user", "type": "access"}, expires_in_seconds=-60
    )
    before = auth_failures_total.labels(reason="expired_token")._value.get()
    with pytest.raises(JWTExpiredError):
        verify_access_token(token)
    assert auth_failures_total.labels(reason="expired_token")._value.get() == before + 1


def test_concurrent_api_rate_limit():
    from fastapi import FastAPI, Request, Response
    from fastapi.testclient import TestClient
    from slowapi import Limiter
    from slowapi.errors import RateLimitExceeded
    from src.api.dependencies import custom_rate_limit_exceeded_handler

    limiter = Limiter(key_func=lambda: "burst-test", headers_enabled=True)
    app = FastAPI()
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, custom_rate_limit_exceeded_handler)

    @app.get("/burst")
    @limiter.limit("5/minute")
    def endpoint(request: Request, response: Response):
        return {"ok": True}

    with TestClient(app) as client:
        first = client.get("/burst")
        assert first.status_code == 200
        with ThreadPoolExecutor(max_workers=8) as pool:
            responses = list(pool.map(lambda _: client.get("/burst"), range(19)))
    assert sum(r.status_code == 200 for r in responses) == 4
    limited = [r for r in responses if r.status_code == 429]
    assert len(limited) == 15
    assert all(int(r.headers["Retry-After"]) > 0 for r in limited)


@pytest.mark.asyncio
async def test_streaming_disk_failure_closes_and_deletes(tmp_path, monkeypatch):
    from src.utils import file_streaming as module

    upload = MagicMock()
    upload.seek = AsyncMock()
    upload.read = AsyncMock(side_effect=[b"a" * 8, b"b" * 8, b""])
    file = open(tmp_path / "partial", "wb")
    proxy = MagicMock(wraps=file)
    proxy.name = str(tmp_path / "partial")
    proxy.write.side_effect = [8, OSError("disk full")]
    monkeypatch.setattr(module.tempfile, "NamedTemporaryFile", lambda **kw: proxy)
    with pytest.raises(module.HTTPException) as exc:
        await module.stream_upload_file_to_disk(upload, chunk_size=8)
    assert exc.value.status_code == 500
    assert file.closed
    assert not (tmp_path / "partial").exists()
    assert all(call.args == (8,) for call in upload.read.call_args_list)


def test_temp_cleanup_logs_and_retains_failure(tmp_path, monkeypatch, caplog):
    from src.utils import temp_manager as module

    directory = tmp_path / "locked"
    directory.mkdir()

    def failing_rmtree(path, **kwargs):
        callback = kwargs.get("onexc") or kwargs.get("onerror")
        callback(None, str(directory / "file"), PermissionError("locked"))

    monkeypatch.setattr(module.shutil, "rmtree", failing_rmtree)
    module.register_temp_path(str(directory))
    module.cleanup_registered_temp_paths()
    assert "locked" in caplog.text


@pytest.mark.parametrize("verified", [True, False])
def test_github_public_email_must_be_verified(monkeypatch, verified):
    from src.utils import sso

    monkeypatch.setenv("GITHUB_CLIENT_ID", "test-id")
    monkeypatch.setenv("GITHUB_CLIENT_SECRET", "test-secret")

    def response(body):
        result = MagicMock(status_code=200, ok=True)
        result.json.return_value = body
        return result

    session = MagicMock()
    session.post.return_value = response({"access_token": "test-token"})
    session.get.side_effect = [
        response({"email": "student@example.edu", "login": "student"}),
        response(
            [{"email": "student@example.edu", "primary": True, "verified": verified}]
        ),
    ]
    monkeypatch.setattr(sso, "_get_oauth_session", lambda: session)
    if verified:
        profile, error = sso.exchange_github_code("test-code")
        assert profile.email == "student@example.edu"
        assert error is None
    else:
        with pytest.raises(ValueError, match="verified"):
            sso.exchange_github_code("test-code")
    assert session.get.call_count == 2


def test_corpus_ingestion_is_atomic_and_idempotent(mock_db, monkeypatch, tmp_path):
    from src.core import corpus_ingestion as module
    from src.db.corpus_db import get_total_document_count, get_chunk_registry

    monkeypatch.setattr(
        module, "get_all_embeddings", lambda: np.ones((1, 384), dtype=np.float32)
    )
    files = {"sample.txt": b"original content"}
    chunks = {"sample.txt": ["original content"]}
    embeddings = {"sample.txt": np.ones((1, 384), dtype=np.float32)}
    index_path = tmp_path / "corpus.index"
    assert module.persist_analyzed_documents(
        files, chunks, embeddings, owner="teacher", index_path=index_path
    )
    assert (
        module.persist_analyzed_documents(
            files, chunks, embeddings, owner="teacher", index_path=index_path
        )
        == {}
    )
    assert get_total_document_count() == 1
    assert len(get_chunk_registry()) == 1
    assert index_path.exists()
    before = index_path.read_bytes()
    with pytest.raises(ValueError, match="Invalid embeddings"):
        module.persist_analyzed_documents(
            {"new.txt": b"new", "bad.txt": b"bad"},
            {"new.txt": ["new"], "bad.txt": ["bad"]},
            {"new.txt": embeddings["sample.txt"], "bad.txt": np.array([[np.nan]])},
            owner="teacher",
            index_path=index_path,
        )
    assert get_total_document_count() == 1
    assert index_path.read_bytes() == before


def test_private_health_routes_are_not_public():
    from src.api.middleware import _is_public_path

    assert _is_public_path("/api/v1/health")
    assert not _is_public_path("/api/v1/health/scores")
    assert not _is_public_path("/api/v1/health/score")


def test_admin_requires_explicit_bootstrap_password(monkeypatch, tmp_path):
    from src.db import auth

    monkeypatch.setattr(auth, "_DB_PATH", str(tmp_path / "auth.db"))
    monkeypatch.delenv("ADMIN_BOOTSTRAP_PASSWORD", raising=False)
    auth.init_db()
    with auth._connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0


def test_auth_upgrade_preserves_custom_columns_and_references():
    import sqlite3
    from src.db.migrations.auth import migrate_auth_database

    with sqlite3.connect(":memory:") as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        migrate_auth_database(conn)
        conn.execute("ALTER TABLE users ADD COLUMN is_active INTEGER DEFAULT 1")
        conn.execute("ALTER TABLE users ADD COLUMN custom_note TEXT")
        conn.execute(
            "INSERT INTO users (username, password, custom_note) VALUES ('existing', 'hash', 'keep')"
        )
        conn.execute(
            "CREATE TABLE linked (user_id INTEGER REFERENCES users(id) ON DELETE CASCADE)"
        )
        conn.execute("INSERT INTO linked SELECT id FROM users")
        conn.execute("PRAGMA user_version = 16")
        conn.commit()
        migrate_auth_database(conn)
        assert conn.execute("SELECT custom_note FROM users").fetchone()[0] == "keep"
        assert conn.execute("SELECT COUNT(*) FROM linked").fetchone()[0] == 1


def test_api_login_verifies_password_and_issues_scoped_tokens(mock_db):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from src.api.dependencies import limiter
    from src.api.routers.auth import router
    from src.db.auth import add_user
    from src.security.jwt_utils import verify_access_token, verify_refresh_token

    add_user("api_teacher", "Teacher-Password!493", role="teacher")
    app = FastAPI()
    app.state.limiter = limiter
    app.include_router(router)
    with TestClient(app) as client:
        assert client.post("/auth/login", json={}).status_code == 422
        assert (
            client.post(
                "/auth/login", json={"username": "api_teacher", "password": "wrong"}  # pragma: allowlist secret -- synthetic test credential
            ).status_code
            == 401
        )
        response = client.post(
            "/auth/login",
            json={"username": "api_teacher", "password": "Teacher-Password!493"},  # pragma: allowlist secret -- synthetic test credential
        )
    assert response.status_code == 200
    payload = verify_access_token(response.json()["token"])
    assert payload["sub"] == "api_teacher"
    assert "scan" in payload["scopes"] and "admin" not in payload["scopes"]
    assert (
        verify_refresh_token(response.json()["refresh_token"])["sub"] == "api_teacher"
    )


def test_api_login_requires_second_factor(mock_db):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from src.api.dependencies import limiter
    from src.api.routers.auth import router
    from src.db.auth import add_user, enable_2fa
    import pyotp

    add_user("two_factor", "Two-Factor-Password!492", role="teacher")
    secret = pyotp.random_base32()
    enable_2fa("two_factor", secret)
    app = FastAPI()
    app.state.limiter = limiter
    app.include_router(router)
    with TestClient(app) as client:
        credentials = {"username": "two_factor", "password": "Two-Factor-Password!492"}  # pragma: allowlist secret -- synthetic test credential
        assert client.post("/auth/login", json=credentials).status_code == 401
        credentials["otp_code"] = pyotp.TOTP(secret).now()
        assert client.post("/auth/login", json=credentials).status_code == 200


def test_api_2fa_setup_requires_password_and_never_returns_existing_secret(mock_db):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from src.api.dependencies import limiter
    from src.api.routers.auth import router
    from src.db.auth import add_user, get_2fa_status

    add_user("enroll_user", "Enroll-Password!493", role="teacher")
    app = FastAPI()
    app.state.limiter = limiter
    app.include_router(router)
    with TestClient(app) as client:
        assert (
            client.post("/auth/2fa/setup", json={"username": "enroll_user"}).status_code
            == 422
        )
        assert (
            client.post(
                "/auth/2fa/setup", json={"username": "enroll_user", "password": "wrong"}  # pragma: allowlist secret -- synthetic test credential
            ).status_code
            == 401
        )
        assert get_2fa_status("enroll_user")[0] is False
        credentials = {"username": "enroll_user", "password": "Enroll-Password!493"}  # pragma: allowlist secret -- synthetic test credential
        first = client.post("/auth/2fa/setup", json=credentials)
        assert first.status_code == 200
        assert get_2fa_status("enroll_user") == (True, first.json()["secret"])
        second = client.post("/auth/2fa/setup", json=credentials)
        assert second.status_code == 409
        assert first.json()["secret"] not in second.text


def test_enabling_2fa_cannot_create_an_administrator(mock_db):
    from src.db.auth import enable_2fa, _connect

    with pytest.raises(ValueError, match="missing user"):
        enable_2fa("unknown_user", "JBSWY3DPEHPK3PXP")
    with _connect() as conn:
        assert (
            conn.execute(
                "SELECT 1 FROM users WHERE username = ?", ("unknown_user",)
            ).fetchone()
            is None
        )


def test_api_refresh_rejects_revoked_token(mock_db):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from src.api.routers.auth import router
    from src.db.auth import add_user, revoke_token
    from src.security.jwt_utils import create_refresh_token

    add_user("refresh_user", "Refresh-Password!493", role="teacher")
    token = create_refresh_token(sub="refresh_user", scopes=["read"])
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        assert (
            client.post(
                "/api/v1/auth/refresh", json={"refresh_token": token}
            ).status_code
            == 200
        )
        revoke_token(token)
        assert (
            client.post(
                "/api/v1/auth/refresh", json={"refresh_token": token}
            ).status_code
            == 401
        )
