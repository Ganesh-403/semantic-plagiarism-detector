"""Applicability assessments must never become blanket vulnerability exceptions."""

import copy
from datetime import date

import pytest

from scripts.audit_dependencies import evaluate, source_evidence


@pytest.fixture
def assessment(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src/inference.py").write_text("from nltk import word_tokenize\nword_tokenize('text')\n")
    return {
        "package": "nltk", "version": "3.10.3", "vulnerability": "GHSA-reviewed",
        "status": "not_affected", "review_by": "2026-10-09", "justification": "Inference only",
        "source_sha256": source_evidence(tmp_path, "nltk"),
    }


@pytest.fixture
def report():
    return {"dependencies": [{"name": "nltk", "version": "3.10.3", "vulns": [
        {"id": "PYSEC-reviewed", "aliases": ["GHSA-reviewed"], "fix_versions": []}
    ]}]}


def test_exact_reviewed_finding_is_retained_as_not_affected(tmp_path, assessment, report):
    original = copy.deepcopy(report)
    blocked, reviewed = evaluate(report, [assessment], root=tmp_path, today=date(2026, 9, 9))
    assert not blocked and len(reviewed) == 1
    assert report == original


@pytest.mark.parametrize("change", ["version", "id", "fix", "expired", "source", "new_source"])
def test_reassessment_required(tmp_path, assessment, report, change):
    today = date(2026, 9, 9)
    dependency = report["dependencies"][0]
    if change == "version":
        dependency["version"] = "3.10.4"
    elif change == "id":
        dependency["vulns"][0] = {"id": "GHSA-unreviewed", "aliases": []}
    elif change == "fix":
        dependency["vulns"][0]["fix_versions"] = ["3.10.4"]
    elif change == "expired":
        today = date(2026, 10, 10)
    elif change == "source":
        (tmp_path / "src/inference.py").write_text("import nltk\nnltk.model.load(user_path)\n")
    else:
        (tmp_path / "src/new.py").write_text("import nltk\n")
    blocked, reviewed = evaluate(report, [assessment], root=tmp_path, today=today)
    assert len(blocked) == 1 and not reviewed


def test_clean_report_and_unresolved_packages(tmp_path):
    assert evaluate({"dependencies": [{"name": "safe", "vulns": []}]}, [], root=tmp_path) == ([], [])
    assert evaluate({"dependencies": []}, [], root=tmp_path)[0]
    assert evaluate({"dependencies": [{"name": "unknown", "skip_reason": "unavailable"}]}, [], root=tmp_path)[0]
