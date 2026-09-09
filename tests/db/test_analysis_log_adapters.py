"""Persist and read each analysis adapter through real, isolated SQLite files."""

from dataclasses import dataclass
from contextlib import closing
from datetime import datetime
import hashlib
import importlib
import json
import sqlite3

import pytest


@dataclass(frozen=True)
class LogCase:
    module: str
    initializer: str
    writer: str
    table: str
    args: tuple
    expected: dict


A = "Essay α '); DROP TABLE data; --"
B = "Comparison β"
CASES = [
    LogCase(
        "api_graph",
        "initialize_api_graph_db",
        "log_api_graph_alignment",
        "api_graph_logs",
        (A, B, 0.75, True),
        {"code_a_id": A, "code_b_id": B, "overall_score": 0.75, "is_clone": 1},
    ),
    LogCase(
        "audio_prosody",
        "initialize_audio_prosody_db",
        "log_prosody_analysis",
        "audio_prosody_logs",
        (A, False, 0.25),
        {"document_id": A, "is_synthetic": 0, "pause_variance": 0.25},
    ),
    LogCase(
        "cad_plagiarism",
        "initialize_cad_plagiarism_db",
        "log_cad_alignment",
        "cad_plagiarism_logs",
        (A, B, 0.75, True),
        {
            "model_a_id": A,
            "model_b_id": B,
            "overall_score": 0.75,
            "is_cloned_geometry": 1,
        },
    ),
    LogCase(
        "cfg_logs",
        "initialize_cfg_db",
        "log_cfg_comparison",
        "cfg_logs",
        (A, B, "hash-a", "hash-b", 2, 0.75, False),
        {
            "document_a_id": A,
            "document_b_id": B,
            "cfg_hash_a": "hash-a",
            "cfg_hash_b": "hash-b",
            "edit_distance": 2,
            "structural_similarity": 0.75,
            "is_exact_clone": 0,
        },
    ),
    LogCase(
        "citation_context",
        "initialize_citation_context_db",
        "log_citation_alignment",
        "citation_context_logs",
        (A, "[1]", 0.75, False),
        {
            "document_id": A,
            "citation_id": "[1]",
            "alignment_score": 0.75,
            "is_bluffing": 0,
        },
    ),
    LogCase(
        "code_comment",
        "initialize_code_comment_db",
        "log_code_comment_alignment",
        "code_comment_logs",
        (A, 0.75, True),
        {"document_id": A, "overall_coherence": 0.75, "is_mismatch": 1},
    ),
    LogCase(
        "cognitive_load",
        "initialize_cognitive_load_db",
        "log_cognitive_load_analysis",
        "cognitive_load_logs",
        (A, 0.75, True, 0.25),
        {
            "document_id": A,
            "ai_probability": 0.75,
            "is_ai_generated": 1,
            "fk_variance": 0.25,
        },
    ),
    LogCase(
        "concept_graphs",
        "initialize_concept_graphs_db",
        "log_concept_alignment",
        "concept_graph_logs",
        (A, B, 0.75, True),
        {
            "doc_a_id": A,
            "doc_b_id": B,
            "conceptual_score": 0.75,
            "is_conceptual_plagiarism": 1,
        },
    ),
    LogCase(
        "cross_modal_logs",
        "initialize_cross_modal_logs_db",
        "log_cross_modal_alignment",
        "cross_modal_logs",
        (A, B, 0.75, False),
        {
            "text_doc_id": A,
            "code_doc_id": B,
            "overall_score": 0.75,
            "is_translation": 0,
        },
    ),
    LogCase(
        "discourse_trees",
        "initialize_discourse_trees_db",
        "log_discourse_alignment",
        "discourse_tree_logs",
        (A, B, 0.75, False),
        {
            "doc_a_id": A,
            "doc_b_id": B,
            "structural_similarity": 0.75,
            "is_structural_plagiarism": 0,
        },
    ),
    LogCase(
        "drift_alerts",
        "initialize_drift_db",
        "log_drift_alert",
        "drift_alerts",
        (A, [{"confidence": 0.2}, {"confidence": 0.9}, {}]),
        {
            "document_id": A,
            "changepoints_json": [{"confidence": 0.2}, {"confidence": 0.9}, {}],
            "max_confidence": 0.9,
        },
    ),
    LogCase(
        "essay_scores",
        "initialize_essay_scores_db",
        "log_essay_score",
        "essay_scores",
        (
            A,
            "rubric α",
            82.5,
            {"clarity": 0.8},
            [{"criterion": "evidence", "score": 3}],
        ),
        {
            "document_id": A,
            "rubric_name": "rubric α",
            "final_grade": 82.5,
            "traits_json": {"clarity": 0.8},
            "criterion_scores_json": [{"criterion": "evidence", "score": 3}],
        },
    ),
    LogCase(
        "evasion_logs",
        "initialize_evasion_logs_db",
        "log_evasion_analysis",
        "evasion_logs",
        (A, 0.75, True, ["pattern α", "quoted ' pattern"]),
        {
            "document_id": A,
            "evasion_risk_score": 0.75,
            "is_suspicious": 1,
            "evasion_patterns": ["pattern α", "quoted ' pattern"],
        },
    ),
    LogCase(
        "git_forensics",
        "initialize_git_forensics_db",
        "log_git_forensics",
        "git_forensics_logs",
        (A, B, 0.75, True),
        {
            "log_a_id": A,
            "log_b_id": B,
            "overall_score": 0.75,
            "is_covert_collaboration": 1,
        },
    ),
    LogCase(
        "layout_logs",
        "initialize_layout_logs_db",
        "log_layout_comparison",
        "layout_logs",
        (A, B, 3, 0.75, False),
        {
            "document_a_id": A,
            "document_b_id": B,
            "edit_distance": 3,
            "structural_similarity": 0.75,
            "is_structural_clone": 0,
        },
    ),
    LogCase(
        "math_plagiarism",
        "initialize_math_plagiarism_db",
        "log_math_alignment",
        "math_plagiarism_logs",
        (A, B, 0.75, True),
        {
            "eq_a_id": A,
            "eq_b_id": B,
            "structural_similarity": 0.75,
            "is_structural_plagiarism": 1,
        },
    ),
    LogCase(
        "multimedia_forensics",
        "initialize_multimedia_forensics_db",
        "log_av_forensics",
        "multimedia_forensics_logs",
        (A, B, 0.75, False),
        {"media_a_id": A, "media_b_id": B, "dubbing_probability": 0.75, "is_dubbed": 0},
    ),
    LogCase(
        "multimodal_corpus",
        "initialize_multimodal_db",
        "store_image_hash",
        "image_hashes",
        (A, B, "hash-value"),
        {"document_id": A, "image_id": B, "phash": "hash-value"},
    ),
    LogCase(
        "notebook_lineage",
        "initialize_notebook_lineage_db",
        "log_notebook_alignment",
        "notebook_lineage_logs",
        (A, B, 0.75, True),
        {"nb_a_id": A, "nb_b_id": B, "overall_score": 0.75, "is_cloned_workflow": 1},
    ),
    LogCase(
        "ocr_extractions",
        "initialize_ocr_extractions_db",
        "log_ocr_extraction",
        "ocr_extraction_logs",
        (A, "image-hash", 3, 0.75, "Extracted α text"),
        {
            "document_id": A,
            "image_hash": "image-hash",
            "block_count": 3,
            "layout_coherence": 0.75,
            "extracted_text_hash": hashlib.sha256(
                "Extracted α text".encode()
            ).hexdigest(),
        },
    ),
    LogCase(
        "patchwriting_logs",
        "initialize_patchwriting_db",
        "log_patchwriting_detection",
        "patchwriting_logs",
        (A, B, 0.75, 0.5, True),
        {
            "document_a_id": A,
            "document_b_id": B,
            "syntactic_jaccard": 0.75,
            "ngram_overlap": 0.5,
            "is_flagged": 1,
        },
    ),
    LogCase(
        "presentation_plagiarism",
        "initialize_presentation_plagiarism_db",
        "log_presentation_alignment",
        "presentation_plagiarism_logs",
        (A, B, 0.75, False),
        {"deck_a_id": A, "deck_b_id": B, "overall_score": 0.75, "is_cloned_deck": 0},
    ),
    LogCase(
        "provenance_logs",
        "initialize_provenance_db",
        "log_provenance_analysis",
        "provenance_logs",
        (A, "pdf", 0.75, True, {"author": "α"}),
        {
            "document_id": A,
            "file_type": "pdf",
            "risk_score": 0.75,
            "is_suspicious": 1,
            "metadata_json": {"author": "α"},
        },
    ),
    LogCase(
        "reviewer_calibration",
        "initialize_calibration_db",
        "log_review_override",
        "review_overrides",
        (A, B, 0.75, 0.5),
        {
            "reviewer_id": A,
            "document_id": B,
            "automated_score": 0.75,
            "manual_score": 0.5,
        },
    ),
    LogCase(
        "revision_bursts",
        "initialize_revision_bursts_db",
        "log_revision_burst_analysis",
        "revision_burst_logs",
        (A, 0.75, True, 0.25),
        {
            "document_id": A,
            "risk_score": 0.75,
            "is_ghostwritten": 1,
            "burst_ratio": 0.25,
        },
    ),
    LogCase(
        "sandbox_logs",
        "initialize_sandbox_db",
        "log_execution_trace",
        "execution_traces",
        (A, B, {"stdout": "α", "exit": 0}),
        {
            "submission_hash": A,
            "behavioral_hash": B,
            "trace_json": {"stdout": "α", "exit": 0},
        },
    ),
    LogCase(
        "semantic_roles",
        "initialize_semantic_roles_db",
        "log_semantic_role_alignment",
        "semantic_role_logs",
        (A, B, 0.75, False),
        {
            "doc_a_id": A,
            "doc_b_id": B,
            "structural_similarity": 0.75,
            "is_deep_paraphrase": 0,
        },
    ),
    LogCase(
        "sql_plagiarism",
        "initialize_sql_plagiarism_db",
        "log_sql_alignment",
        "sql_plagiarism_logs",
        (A, B, 0.75, True),
        {"query_a_id": A, "query_b_id": B, "overall_score": 0.75, "is_cloned_logic": 1},
    ),
    LogCase(
        "steganography_logs",
        "initialize_steganography_logs_db",
        "log_steganography_analysis",
        "steganography_logs",
        (A, True, 0.75, ["pattern α"]),
        {
            "document_id": A,
            "is_injection": 1,
            "risk_score": 0.75,
            "matched_patterns": ["pattern α"],
        },
    ),
    LogCase(
        "tabular_plagiarism",
        "initialize_tabular_plagiarism_db",
        "log_table_alignment",
        "tabular_plagiarism_logs",
        (A, B, 0.75, False),
        {
            "table_a_id": A,
            "table_b_id": B,
            "overall_score": 0.75,
            "is_cloned_dataset": 0,
        },
    ),
    LogCase(
        "template_fingerprints",
        "initialize_template_fingerprints_db",
        "log_template_comparison",
        "template_fingerprint_logs",
        (A, B, True, 0.25),
        {
            "doc_a_id": A,
            "doc_b_id": B,
            "is_template_plagiarism": 1,
            "entropy_delta": 0.25,
        },
    ),
    LogCase(
        "tool_signatures",
        "initialize_signatures_db",
        "log_attribution",
        "attribution_logs",
        (A, "tool α", 0.75, {"entropy": 0.5}),
        {
            "document_id": A,
            "attributed_tool": "tool α",
            "confidence": 0.75,
            "fingerprint_json": {"entropy": 0.5},
        },
    ),
    LogCase(
        "translation_attack_logs",
        "initialize_translation_logs_db",
        "log_translation_attack",
        "translation_attack_logs",
        (A, 0.75, 0.5, 0.25, True),
        {
            "document_id": A,
            "lexical_drift": 0.75,
            "structural_variance": 0.5,
            "invariance_score": 0.25,
            "is_obfuscated": 1,
        },
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.module)
def test_analysis_log_commit_readback_rollback_and_close(case, tmp_path, monkeypatch):
    module = importlib.import_module(f"src.db.{case.module}_db")
    db = tmp_path / "nested" / "analysis.sqlite"
    monkeypatch.setattr(module, "DEFAULT_DB_PATH", db)
    initialize = getattr(module, case.initializer)
    write = getattr(module, case.writer)
    initialize()
    initialize(db_path=db)  # Schema creation is idempotent.
    assert write(*case.args) is True
    with module.get_connection(db) as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert connection.execute("PRAGMA synchronous").fetchone()[0] == 1
        assert connection.execute("PRAGMA busy_timeout").fetchone()[0] == 15000
        records = connection.execute(f'SELECT * FROM "{case.table}"').fetchall()
        assert len(records) == 1
        row = dict(records[0])
        for column, value in case.expected.items():
            stored = (
                json.loads(row[column])
                if isinstance(value, (dict, list))
                else row[column]
            )
            assert stored == value, column
        timestamp = next(value for key, value in row.items() if key.endswith("_at"))
        assert datetime.fromisoformat(timestamp).year >= 2026
    with pytest.raises(sqlite3.ProgrammingError):
        connection.execute("SELECT 1")
    with pytest.raises(RuntimeError, match="rollback"):
        with module.get_connection() as failed:
            failed.execute(f'DELETE FROM "{case.table}"')
            raise RuntimeError("rollback")
    with pytest.raises(sqlite3.ProgrammingError):
        failed.execute("SELECT 1")
    with closing(sqlite3.connect(db)) as reader:
        assert reader.execute(f'SELECT COUNT(*) FROM "{case.table}"').fetchone()[0] == 1
        assert reader.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.module)
def test_analysis_log_missing_schema_reports_failure(case, tmp_path, caplog):
    module = importlib.import_module(f"src.db.{case.module}_db")
    assert (
        getattr(module, case.writer)(*case.args, db_path=tmp_path / "no-schema.db")
        is False
    )
    assert "no such table" in caplog.text


def test_drift_empty_results_and_execution_clone_lookup(tmp_path):
    from src.db import drift_alerts_db as drift, sandbox_logs_db as traces

    db = tmp_path / "drift.db"
    drift.initialize_drift_db(db)
    assert drift.log_drift_alert(A, [], db)
    with drift.get_connection(db) as connection:
        row = connection.execute(
            "SELECT max_confidence, changepoints_json FROM drift_alerts"
        ).fetchone()
        assert tuple(row) == (0, "[]")
    db = tmp_path / "traces.db"
    traces.initialize_sandbox_db(db)
    for submission, behavior in (
        ("one", "same"),
        ("two", "same"),
        ("three", "different"),
    ):
        assert traces.log_execution_trace(
            submission, behavior, {"output": submission}, db
        )
    assert set(traces.find_behavioral_clones("same", db_path=db)) == {"one", "two"}
    assert traces.find_behavioral_clones("same", "one", db) == ["two"]
    assert traces.log_execution_trace("one", "new", {"output": "updated"}, db)
    assert traces.find_behavioral_clones("same", db_path=db) == ["two"]
    assert traces.find_behavioral_clones("absent", db_path=db) == []
    assert traces.find_behavioral_clones("same", db_path=tmp_path / "missing.db") == []


def test_reviewer_metric_updates_and_missing_schema_fallback(tmp_path):
    from src.db import reviewer_calibration_db as module

    db = tmp_path / "reviewers.db"
    module.initialize_calibration_db(db)
    assert module.get_reviewer_weight(A, db) == 1.0
    metrics = {"mean_error": -0.2, "mean_absolute_error": 0.25, "variance": 0.03}
    assert module.update_reviewer_metrics(A, metrics, 0.8, db)
    assert module.get_reviewer_weight(A, db) == 0.8
    assert module.update_reviewer_metrics(A, metrics, 0.7, db)
    with module.get_connection(db) as connection:
        rows = connection.execute("SELECT * FROM reviewer_metrics").fetchall()
        assert len(rows) == 1
        assert rows[0]["mean_error"] == -0.2 and rows[0]["calibration_weight"] == 0.7
    missing = tmp_path / "missing.db"
    assert module.get_reviewer_weight(A, missing) == 1.0
    assert not module.update_reviewer_metrics(A, metrics, 0.8, missing)


def test_federation_rejects_unknown_nodes_and_preserves_registered_signatures(tmp_path):
    from src.db import federation_registry_db as module

    db = tmp_path / "federation.db"
    module.initialize_federation_db(db)
    assert not module.store_federated_signature(A, "unknown", ["band"], db)
    assert module.register_trusted_node("school", "School α", "public-fingerprint", db)
    assert module.store_federated_signature(A, "school", ["band α", "band β"], db)
    assert module.register_trusted_node(
        "school", "Renamed school", "new-fingerprint", db
    )
    with module.get_connection(db) as connection:
        node = connection.execute("SELECT * FROM trusted_nodes").fetchone()
        signature = connection.execute("SELECT * FROM federated_signatures").fetchone()
        assert node["name"] == "Renamed school"
        assert signature["document_id"] == A
        assert json.loads(signature["lsh_bands_json"]) == ["band α", "band β"]
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    assert not module.register_trusted_node(
        "school", "School", "fingerprint", tmp_path / "missing.db"
    )


def test_reviewer_repository_persists_history_and_returns_independent_records(
    tmp_path, monkeypatch
):
    from src.db import reviewer_calibration_db as module

    db_path = tmp_path / "reviews.db"
    monkeypatch.setattr(module, "DEFAULT_DB_PATH", db_path)
    repository = module.ReviewerCalibrationDB()
    assert repository.fetch_all_overrides() == []
    repository.save_review_override(A, "reviewer", 0.9, 0.7)
    repository.save_review_override(B, "other", 0.2, 0.3)
    repository.save_review_override(B, "reviewer", 0.8, 0.7)
    restored = module.ReviewerCalibrationDB(db_path)
    history = restored.fetch_reviewer_history("reviewer")
    assert [record["submission_id"] for record in history] == [A, B]
    assert history[0]["consensus_deviation"] == pytest.approx(0.2)
    assert len(restored.fetch_all_overrides()) == 3
    assert restored.fetch_reviewer_history("missing") == []
    history[0]["assigned_score"] = 0
    assert restored.fetch_reviewer_history("reviewer")[0]["assigned_score"] == 0.9
    for assigned, consensus in ((-1, 0.5), (0.5, 2), (float("nan"), 0.5)):
        with pytest.raises(ValueError):
            repository.save_review_override(A, "reviewer", assigned, consensus)
    with module.get_connection(db_path) as connection:
        connection.execute("DROP TABLE review_overrides")
    with pytest.raises(sqlite3.OperationalError, match="persist"):
        repository.save_review_override(A, "reviewer", 0.9, 0.7)


def test_patchwriting_repository_persists_metrics_and_rolls_back_invalid_json(
    tmp_path, monkeypatch
):
    from src.db import patchwriting_logs_db as module

    db_path = tmp_path / "patchwriting.db"
    monkeypatch.setattr(module, "DEFAULT_DB_PATH", db_path)
    repository = module.PatchwritingLogsDB()
    assert repository.fetch_all_logs() == []
    metrics = {"tokens": ["α", "β"], "overlap": 0.25}
    repository.log_structural_clone(A, B, 0.75, metrics)
    repository.log_structural_clone("other", A, 0.5, {})
    metrics["tokens"].append("changed")
    restored = module.PatchwritingLogsDB(db_path)
    logs = restored.fetch_logs_by_submission(A)
    assert len(logs) == 1
    assert logs[0]["metrics"] == {"tokens": ["α", "β"], "overlap": 0.25}
    assert logs[0]["source_id"] == B and logs[0]["similarity_score"] == 0.75
    assert datetime.fromisoformat(logs[0]["timestamp"])
    assert len(restored.fetch_all_logs()) == 2
    assert restored.fetch_logs_by_submission("missing") == []
    with pytest.raises(ValueError):
        repository.log_structural_clone(A, B, 0.5, {"bad": float("nan")})
    for score in (-1, 2, float("nan")):
        with pytest.raises(ValueError):
            repository.log_structural_clone(A, B, score, {})
    assert len(restored.fetch_all_logs()) == 2
