"""Tests for src.core.analysis_report_generator."""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest

from src.core.analysis_report_generator import (
    AnalysisReport,
    AnalysisReportGenerator,
    PairwiseFinding,
    ScanSummary,
    export_report_html,
    export_report_json,
    export_report_markdown,
)
from src.core.config import HIGH_SEVERITY, LOW_SEVERITY, MEDIUM_SEVERITY, SimilarityThresholds


# ---------------------------------------------------------------------------
# PairwiseFinding
# ---------------------------------------------------------------------------

class TestPairwiseFinding:
    def test_to_dict(self):
        f = PairwiseFinding("a.pdf", "b.pdf", 0.85, HIGH_SEVERITY)
        d = f.to_dict()
        assert d["doc_a"] == "a.pdf"
        assert d["similarity"] == 0.85
        assert d["severity"] == HIGH_SEVERITY

    def test_with_chunk_data(self):
        chunks = [{"uploaded": "text a", "matched": "text b", "score": 0.9}]
        f = PairwiseFinding("a.pdf", "b.pdf", 0.9, HIGH_SEVERITY, flagged_chunks=chunks)
        d = f.to_dict()
        assert len(d["flagged_chunks"]) == 1


# ---------------------------------------------------------------------------
# ScanSummary
# ---------------------------------------------------------------------------

class TestScanSummary:
    def test_to_dict(self):
        s = ScanSummary(
            total_documents=10,
            total_pairs=45,
            flagged_pairs=5,
            avg_similarity=0.45,
            max_similarity=0.95,
            median_similarity=0.42,
            severity_distribution={HIGH_SEVERITY: 2, MEDIUM_SEVERITY: 3},
            threshold_used=0.59,
            scan_timestamp="2025-01-01T00:00:00",
            processing_time_seconds=1.5,
        )
        d = s.to_dict()
        assert d["total_documents"] == 10
        assert d["flagging_rate"] == pytest.approx(5 / 45, rel=1e-4)

    def test_zero_pairs_flagging_rate(self):
        s = ScanSummary(
            total_documents=1,
            total_pairs=0,
            flagged_pairs=0,
            avg_similarity=0.0,
            max_similarity=0.0,
            median_similarity=0.0,
            severity_distribution={},
            threshold_used=0.59,
            scan_timestamp="2025-01-01T00:00:00",
            processing_time_seconds=0.0,
        )
        d = s.to_dict()
        assert d["flagging_rate"] == 0.0


# ---------------------------------------------------------------------------
# AnalysisReportGenerator
# ---------------------------------------------------------------------------

class TestAnalysisReportGenerator:
    def _make_matrix(self, n, high_pairs=None):
        mat = np.full((n, n), 0.3)
        np.fill_diagonal(mat, 1.0)
        for i, j in (high_pairs or []):
            mat[i, j] = 0.95
            mat[j, i] = 0.95
        return mat

    def test_generate_basic(self):
        names = ["a.pdf", "b.pdf", "c.pdf"]
        mat = self._make_matrix(3, [(0, 1)])
        gen = AnalysisReportGenerator()
        report = gen.generate(names, mat)
        assert report.scan_summary.total_documents == 3
        assert len(report.findings) > 0

    def test_generate_empty(self):
        gen = AnalysisReportGenerator()
        report = gen.generate([], np.empty((0, 0)))
        assert report.scan_summary.total_documents == 0
        assert len(report.findings) == 0

    def test_generate_single_doc(self):
        gen = AnalysisReportGenerator()
        report = gen.generate(["a.pdf"], np.eye(1))
        assert report.scan_summary.total_documents == 1

    def test_shape_mismatch_raises(self):
        gen = AnalysisReportGenerator()
        with pytest.raises(ValueError, match="does not match"):
            gen.generate(["a", "b"], np.eye(3))

    def test_to_json(self):
        names = ["a.pdf", "b.pdf"]
        mat = self._make_matrix(2, [(0, 1)])
        report = AnalysisReportGenerator().generate(names, mat)
        json_str = report.to_json()
        parsed = json.loads(json_str)
        assert "scan_summary" in parsed
        assert "findings" in parsed

    def test_to_markdown(self):
        names = ["a.pdf", "b.pdf", "c.pdf"]
        mat = self._make_matrix(3, [(0, 1)])
        report = AnalysisReportGenerator().generate(names, mat)
        md = report.to_markdown()
        assert "# Plagiarism Analysis Report" in md
        assert "a.pdf" in md

    def test_findings_sorted_by_similarity(self):
        names = ["a.pdf", "b.pdf", "c.pdf"]
        mat = np.array([
            [1.0, 0.95, 0.7],
            [0.95, 1.0, 0.8],
            [0.7, 0.8, 1.0],
        ])
        report = AnalysisReportGenerator().generate(names, mat)
        sims = [f.similarity for f in report.findings]
        assert sims == sorted(sims, reverse=True)

    def test_max_findings_limit(self):
        names = ["a.pdf", "b.pdf", "c.pdf", "d.pdf"]
        mat = np.array([
            [1.0, 0.9, 0.85, 0.8],
            [0.9, 1.0, 0.88, 0.75],
            [0.85, 0.88, 1.0, 0.92],
            [0.8, 0.75, 0.92, 1.0],
        ])
        gen = AnalysisReportGenerator(max_findings=2)
        report = gen.generate(names, mat)
        assert len(report.findings) <= 2

    def test_severity_classification(self):
        names = ["a.pdf", "b.pdf"]
        mat = np.array([[1.0, 0.95], [0.95, 1.0]])
        report = AnalysisReportGenerator().generate(names, mat)
        assert report.findings[0].severity == HIGH_SEVERITY

    def test_cluster_analysis_included(self):
        names = ["a.pdf", "b.pdf", "c.pdf", "d.pdf"]
        mat = self._make_matrix(4, [(0, 1), (2, 3)])
        report = AnalysisReportGenerator(enable_clustering=True).generate(names, mat)
        assert report.cluster_result is not None
        assert report.cluster_result.total_clusters >= 1

    def test_cluster_analysis_disabled(self):
        names = ["a.pdf", "b.pdf", "c.pdf"]
        mat = self._make_matrix(3, [(0, 1)])
        report = AnalysisReportGenerator(enable_clustering=False).generate(names, mat)
        assert report.cluster_result is None

    def test_recommendations_generated(self):
        names = ["a.pdf", "b.pdf", "c.pdf"]
        mat = self._make_matrix(3, [(0, 1)])
        report = AnalysisReportGenerator().generate(names, mat)
        assert len(report.recommendations) > 0

    def test_document_risks_computed(self):
        names = ["a.pdf", "b.pdf"]
        mat = np.array([[1.0, 0.88], [0.88, 1.0]])
        report = AnalysisReportGenerator().generate(names, mat)
        assert "a.pdf" in report.document_risks
        assert "max_similarity" in report.document_risks["a.pdf"]

    def test_risk_summary(self):
        names = ["a.pdf", "b.pdf"]
        mat = np.array([[1.0, 0.95], [0.95, 1.0]])
        report = AnalysisReportGenerator().generate(names, mat)
        rs = report.risk_summary
        assert rs["total_documents"] == 2
        assert rs["max_risk_score"] > 0.9

    def test_chunk_level_data_attached(self):
        names = ["a.pdf", "b.pdf"]
        mat = np.array([[1.0, 0.92], [0.92, 1.0]])
        chunk_data = {
            "a.pdf|b.pdf": [
                {"uploaded": "text", "matched": "text", "score": 0.92}
            ]
        }
        report = AnalysisReportGenerator().generate(
            names, mat, chunk_level_data=chunk_data
        )
        assert len(report.findings[0].flagged_chunks) == 1


# ---------------------------------------------------------------------------
# Export functions
# ---------------------------------------------------------------------------

class TestExportFunctions:
    def _make_report(self):
        names = ["a.pdf", "b.pdf", "c.pdf"]
        mat = np.array([
            [1.0, 0.92, 0.3],
            [0.92, 1.0, 0.4],
            [0.3, 0.4, 1.0],
        ])
        return AnalysisReportGenerator().generate(names, mat)

    def test_export_json(self):
        report = self._make_report()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "report.json")
            result = export_report_json(report, path)
            assert os.path.exists(result)
            with open(result) as f:
                data = json.load(f)
            assert "scan_summary" in data

    def test_export_markdown(self):
        report = self._make_report()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "report.md")
            result = export_report_markdown(report, path)
            assert os.path.exists(result)
            with open(result) as f:
                content = f.read()
            assert "Plagiarism Analysis Report" in content

    def test_export_html(self):
        report = self._make_report()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "report.html")
            result = export_report_html(report, path)
            assert os.path.exists(result)
            with open(result) as f:
                content = f.read()
            assert "<!DOCTYPE html>" in content
            assert "a.pdf" in content
