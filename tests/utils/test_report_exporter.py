"""Tests for src/utils/report_exporter.py — Plagiarism Analytics Report Exporter."""

from __future__ import annotations

import csv
import io
import json
import math
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.core.config import DEFAULT_THRESHOLDS, SimilarityThresholds
from src.utils.report_exporter import (
    AnalyticsReport,
    DocumentRiskProfile,
    FlaggedPairRecord,
    ReportAccumulator,
    SeverityDistribution,
    SimilarityStatistics,
    build_analytics_report,
    compute_severity_distribution,
    compute_similarity_statistics,
    export_report_csv_clusters,
    export_report_csv_documents,
    export_report_csv_flags,
    export_report_json,
    format_flagged_pairs,
    format_summary_text,
    generate_report_files,
    rank_document_risk,
    summarize_clusters,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_similarity_df() -> pd.DataFrame:
    """A small 4×4 similarity matrix with realistic scores."""
    data = [
        [1.00, 0.92, 0.45, 0.12],
        [0.92, 1.00, 0.78, 0.33],
        [0.45, 0.78, 1.00, 0.56],
        [0.12, 0.33, 0.56, 1.00],
    ]
    return pd.DataFrame(
        data,
        columns=["essay_a.pdf", "essay_b.pdf", "essay_c.pdf", "essay_d.pdf"],
        index=["essay_a.pdf", "essay_b.pdf", "essay_c.pdf", "essay_d.pdf"],
    )


@pytest.fixture
def sample_flags() -> list[dict]:
    """Flagged-pair dicts that match the similarity matrix."""
    return [
        {"doc_a": "essay_a.pdf", "doc_b": "essay_b.pdf", "similarity": 0.92, "severity": "High"},
        {"doc_a": "essay_b.pdf", "doc_b": "essay_c.pdf", "similarity": 0.78, "severity": "Medium"},
    ]


@pytest.fixture
def sample_clusters() -> dict[int, list[str]]:
    """A cluster with three documents."""
    return {
        0: ["essay_a.pdf", "essay_b.pdf", "essay_c.pdf"],
    }


# ---------------------------------------------------------------------------
# SimilarityStatistics
# ---------------------------------------------------------------------------


class TestSimilarityStatistics:
    """Tests for compute_similarity_statistics()."""

    def test_basic_computation(self, sample_similarity_df: pd.DataFrame):
        stats = compute_similarity_statistics(sample_similarity_df)
        assert stats.count > 0
        assert 0.0 <= stats.mean <= 1.0
        assert 0.0 <= stats.median <= 1.0
        assert stats.min_score <= stats.mean <= stats.max_score

    def test_off_diagonal_only(self):
        """The diagonal (self-similarity = 1.0) should not skew the stats."""
        df = pd.DataFrame(
            [[1.0, 0.2], [0.2, 1.0]],
            columns=["A", "B"],
            index=["A", "B"],
        )
        stats = compute_similarity_statistics(df)
        assert stats.mean == pytest.approx(0.2, abs=1e-6)
        assert stats.max_score == pytest.approx(0.2, abs=1e-6)

    def test_empty_input(self):
        stats = compute_similarity_statistics(pd.DataFrame())
        assert stats.count == 0
        assert stats.mean == 0.0

    def test_single_value(self):
        stats = compute_similarity_statistics([0.75])
        assert stats.count == 1
        assert stats.mean == pytest.approx(0.75, abs=1e-6)
        assert stats.min_score == stats.max_score == pytest.approx(0.75)

    def test_list_input(self):
        scores = [0.1, 0.2, 0.3, 0.4, 0.5]
        stats = compute_similarity_statistics(scores)
        assert stats.count == 5
        assert stats.mean == pytest.approx(0.3, abs=1e-6)
        assert stats.min_score == pytest.approx(0.1)
        assert stats.max_score == pytest.approx(0.5)

    def test_nan_infinity_handled(self):
        stats = compute_similarity_statistics([0.5, float("nan"), float("inf"), -float("inf")])
        assert stats.count == 1  # only 0.5 survives
        assert stats.mean == pytest.approx(0.5)

    def test_percentile_ordering(self, sample_similarity_df: pd.DataFrame):
        stats = compute_similarity_statistics(sample_similarity_df)
        assert stats.p25 <= stats.median <= stats.p75 <= stats.p90 <= stats.p95

    def test_to_dict_returns_all_keys(self):
        stats = compute_similarity_statistics([0.1, 0.9])
        d = stats.to_dict()
        assert "count" in d
        assert "mean" in d
        assert "p95" in d


# ---------------------------------------------------------------------------
# SeverityDistribution
# ---------------------------------------------------------------------------


class TestSeverityDistribution:
    """Tests for compute_severity_distribution()."""

    def test_basic_distribution(self, sample_similarity_df: pd.DataFrame):
        dist = compute_severity_distribution(sample_similarity_df)
        assert dist.total > 0
        assert dist.low_count + dist.medium_count + dist.high_count == dist.total
        assert dist.flagged_count == dist.medium_count + dist.high_count

    def test_percentage_sums_to_100(self, sample_similarity_df: pd.DataFrame):
        dist = compute_severity_distribution(sample_similarity_df)
        total_pct = dist.low_pct + dist.medium_pct + dist.high_pct
        assert total_pct == pytest.approx(100.0, abs=0.1)

    def test_all_zero_matrix(self):
        df = pd.DataFrame(
            [[0.0, 0.0], [0.0, 0.0]],
            columns=["A", "B"],
            index=["A", "B"],
        )
        dist = compute_severity_distribution(df)
        assert dist.flagged_count == 0
        assert dist.high_count == 0

    def test_custom_threshold(self, sample_similarity_df: pd.DataFrame):
        thr = SimilarityThresholds(plagiarism=0.50, medium=0.70, high=0.90)
        dist = compute_severity_distribution(sample_similarity_df, thr)
        # More pairs should be flagged with a lower plagiarism threshold
        assert dist.flagged_count >= 1


# ---------------------------------------------------------------------------
# Document risk ranking
# ---------------------------------------------------------------------------


class TestDocumentRiskRanking:
    """Tests for rank_document_risk()."""

    def test_returns_profiles_for_all_documents(self, sample_similarity_df: pd.DataFrame):
        profiles = rank_document_risk(sample_similarity_df)
        assert len(profiles) == sample_similarity_df.shape[0]

    def test_sorted_by_risk_descending(self, sample_similarity_df: pd.DataFrame):
        profiles = rank_document_risk(sample_similarity_df)
        max_sims = [p.max_similarity for p in profiles]
        assert max_sims == sorted(max_sims, reverse=True)

    def test_risk_level_labels(self):
        """Documents with high similarity get higher risk labels."""
        data = [
            [1.0, 0.95, 0.10],
            [0.95, 1.0, 0.10],
            [0.10, 0.10, 1.0],
        ]
        df = pd.DataFrame(data, columns=["A", "B", "C"], index=["A", "B", "C"])
        profiles = rank_document_risk(df)
        risk_map = {p.document_name: p.risk_level for p in profiles}
        assert risk_map["A"] in ("Critical", "High")
        assert risk_map["C"] == "Low"

    def test_to_dict(self, sample_similarity_df: pd.DataFrame):
        profiles = rank_document_risk(sample_similarity_df)
        for p in profiles:
            d = p.to_dict()
            assert "document_name" in d
            assert "risk_level" in d


# ---------------------------------------------------------------------------
# Cluster summaries
# ---------------------------------------------------------------------------


class TestClusterSummaries:
    """Tests for summarize_clusters()."""

    def test_basic_cluster_summary(self, sample_similarity_df: pd.DataFrame, sample_clusters: dict):
        summaries = summarize_clusters(sample_clusters, sample_similarity_df)
        assert len(summaries) == 1
        s = summaries[0]
        assert s.member_count == 3
        assert s.avg_internal_similarity > 0.0

    def test_empty_clusters(self, sample_similarity_df: pd.DataFrame):
        summaries = summarize_clusters({}, sample_similarity_df)
        assert len(summaries) == 0

    def test_single_member_skipped(self, sample_similarity_df: pd.DataFrame):
        summaries = summarize_clusters({0: ["A"]}, sample_similarity_df)
        assert len(summaries) == 0

    def test_to_dict(self, sample_similarity_df: pd.DataFrame, sample_clusters: dict):
        summaries = summarize_clusters(sample_clusters, sample_similarity_df)
        d = summaries[0].to_dict()
        assert "cluster_id" in d
        assert "documents" in d


# ---------------------------------------------------------------------------
# Flagged pair formatting
# ---------------------------------------------------------------------------


class TestFlaggedPairFormatting:
    """Tests for format_flagged_pairs()."""

    def test_basic_formatting(self, sample_flags: list[dict]):
        records = format_flagged_pairs(sample_flags)
        assert len(records) == 2
        assert records[0].similarity >= records[1].similarity

    def test_empty_input(self):
        records = format_flagged_pairs([])
        assert records == []

    def test_alternative_keys(self):
        flags = [{"document_a": "X", "document_b": "Y", "similarity_score": 0.88}]
        records = format_flagged_pairs(flags)
        assert len(records) == 1
        assert records[0].doc_a == "X"

    def test_to_dict(self, sample_flags: list[dict]):
        records = format_flagged_pairs(sample_flags)
        d = records[0].to_dict()
        assert "doc_a" in d
        assert "severity" in d

def test_format_flagged_pairs_preserves_hybrid_decomposition():
    from src.utils.report_exporter import format_flagged_pairs

    flags = [
        {
            "doc_a": "document-a",
            "doc_b": "document-b",
            "hybrid_score": 0.84,
            "semantic_score": 0.90,
            "lexical_score": 0.70,
            "semantic_contribution": 0.63,
            "lexical_contribution": 0.21,
            "alpha": 0.70,
            "threshold": 0.59,
            "threshold_margin": 0.25,
            "severity": "Medium",
        }
    ]

    records = format_flagged_pairs(flags)

    assert len(records) == 1

    record = records[0]

    assert record.semantic_score == 0.90
    assert record.lexical_score == 0.70
    assert record.semantic_contribution == 0.63
    assert record.lexical_contribution == 0.21
    assert record.hybrid_weight == 0.70
    assert record.threshold_margin == 0.25
    assert record.severity == "Medium"
# ---------------------------------------------------------------------------
# Build analytics report
# ---------------------------------------------------------------------------


class TestBuildAnalyticsReport:
    """Tests for the top-level build_analytics_report()."""

    def test_full_report(
        self,
        sample_similarity_df: pd.DataFrame,
        sample_flags: list[dict],
        sample_clusters: dict,
    ):
        report = build_analytics_report(
            sample_similarity_df,
            flagged_pairs=sample_flags,
            clusters=sample_clusters,
            metadata={"model": "test-model"},
        )
        assert report.total_documents == 4
        assert report.total_evaluated_pairs == 6
        assert len(report.flagged_pairs) == 2
        assert len(report.cluster_summaries) == 1
        assert report.metadata["model"] == "test-model"

    def test_minimal_report(self, sample_similarity_df: pd.DataFrame):
        report = build_analytics_report(sample_similarity_df)
        assert report.total_documents == 4
        assert report.flagged_pairs == []
        assert report.cluster_summaries == []

    def test_custom_thresholds(self, sample_similarity_df: pd.DataFrame):
        thr = SimilarityThresholds(plagiarism=0.30, medium=0.50, high=0.80)
        report = build_analytics_report(sample_similarity_df, thresholds=thr)
        assert report.threshold == 0.30
        assert report.severity_distribution.flagged_count >= 1

    def test_to_dict_roundtrip(self, sample_similarity_df: pd.DataFrame, sample_flags: list[dict]):
        report = build_analytics_report(sample_similarity_df, flagged_pairs=sample_flags)
        d = report.to_dict()
        assert "similarity_statistics" in d
        assert "severity_distribution" in d
        assert "document_risk_profiles" in d
        assert isinstance(d["document_risk_profiles"], list)


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------


class TestJsonExport:
    """Tests for export_report_json()."""

    def test_valid_json(self, sample_similarity_df: pd.DataFrame):
        report = build_analytics_report(sample_similarity_df)
        json_str = export_report_json(report)
        parsed = json.loads(json_str)
        assert "generated_at" in parsed
        assert "similarity_statistics" in parsed

    def test_empty_report_json(self):
        df = pd.DataFrame(
            [[1.0, 0.0], [0.0, 1.0]],
            columns=["A", "B"],
            index=["A", "B"],
        )
        report = build_analytics_report(df)
        json_str = export_report_json(report)
        parsed = json.loads(json_str)
        assert parsed["total_documents"] == 2

    def test_indentation(self, sample_similarity_df: pd.DataFrame):
        report = build_analytics_report(sample_similarity_df)
        json_str = export_report_json(report, indent=4)
        assert "    " in json_str  # 4-space indent present


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------


class TestCsvExport:
    """Tests for CSV export functions."""

    def _read_csv(self, csv_str: str) -> list[dict]:
        reader = csv.DictReader(io.StringIO(csv_str))
        return list(reader)

    def test_flags_csv(self, sample_similarity_df: pd.DataFrame, sample_flags: list[dict]):
        report = build_analytics_report(sample_similarity_df, flagged_pairs=sample_flags)
        csv_str = export_report_csv_flags(report)
        rows = self._read_csv(csv_str)
        assert len(rows) == 2
        assert "doc_a" in rows[0]
        assert "similarity" in rows[0]

    def test_documents_csv(self, sample_similarity_df: pd.DataFrame):
        report = build_analytics_report(sample_similarity_df)
        csv_str = export_report_csv_documents(report)
        rows = self._read_csv(csv_str)
        assert len(rows) == 4
        assert "risk_level" in rows[0]

    def test_clusters_csv(self, sample_similarity_df: pd.DataFrame, sample_clusters: dict):
        report = build_analytics_report(sample_similarity_df, clusters=sample_clusters)
        csv_str = export_report_csv_clusters(report)
        rows = self._read_csv(csv_str)
        assert len(rows) == 1
        assert "cluster_id" in rows[0]

    def test_empty_flags_csv(self, sample_similarity_df: pd.DataFrame):
        report = build_analytics_report(sample_similarity_df)
        csv_str = export_report_csv_flags(report)
        rows = self._read_csv(csv_str)
        assert len(rows) == 0


# ---------------------------------------------------------------------------
# File generation
# ---------------------------------------------------------------------------


class TestGenerateReportFiles:
    """Tests for generate_report_files()."""

    def test_creates_all_files(self, sample_similarity_df: pd.DataFrame, sample_flags: list[dict]):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = generate_report_files(
                sample_similarity_df,
                flagged_pairs=sample_flags,
                output_dir=tmpdir,
            )
            assert "json" in paths
            assert "flags_csv" in paths
            assert "documents_csv" in paths
            assert "clusters_csv" in paths
            for key, path in paths.items():
                assert os.path.exists(path), f"File for {key} not found at {path}"
                assert os.path.getsize(path) > 0

    def test_json_file_is_valid(self, sample_similarity_df: pd.DataFrame):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = generate_report_files(sample_similarity_df, output_dir=tmpdir)
            with open(paths["json"], "r", encoding="utf-8") as f:
                data = json.load(f)
            assert "similarity_statistics" in data

    def test_creates_output_dir_if_missing(self, sample_similarity_df: pd.DataFrame):
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = os.path.join(tmpdir, "reports", "sub")
            paths = generate_report_files(sample_similarity_df, output_dir=nested)
            assert os.path.exists(paths["json"])


# ---------------------------------------------------------------------------
# Summary text
# ---------------------------------------------------------------------------


class TestFormatSummaryText:
    """Tests for format_summary_text()."""

    def test_basic_summary(self, sample_similarity_df: pd.DataFrame):
        report = build_analytics_report(sample_similarity_df)
        text = format_summary_text(report)
        assert "docs" in text
        assert "median" in text

    def test_empty_report_summary(self):
        df = pd.DataFrame(
            [[1.0, 0.0], [0.0, 1.0]],
            columns=["A", "B"],
            index=["A", "B"],
        )
        report = build_analytics_report(df)
        text = format_summary_text(report)
        assert "2 docs" in text


# ---------------------------------------------------------------------------
# ReportAccumulator (trend tracking)
# ---------------------------------------------------------------------------


class TestReportAccumulator:
    """Tests for the ReportAccumulator trend-tracking class."""

    def _make_report(self, mean_sim: float, flagged: int = 0) -> AnalyticsReport:
        """Helper to build a minimal report with controlled values."""
        data = [[1.0, mean_sim], [mean_sim, 1.0]]
        df = pd.DataFrame(data, columns=["A", "B"], index=["A", "B"])
        flags = [{"doc_a": "A", "doc_b": "B", "similarity": mean_sim}] * flagged
        return build_analytics_report(df, flagged_pairs=flags)

    def test_add_session(self):
        acc = ReportAccumulator()
        acc.add_session(self._make_report(0.5))
        assert acc.session_count == 1

    def test_build_trend_direction_increasing(self):
        acc = ReportAccumulator()
        acc.add_session(self._make_report(0.3))
        acc.add_session(self._make_report(0.6))
        trend = acc.build_trend()
        assert trend["aggregate"]["trend_direction"] == "increasing"

    def test_build_trend_direction_decreasing(self):
        acc = ReportAccumulator()
        acc.add_session(self._make_report(0.7))
        acc.add_session(self._make_report(0.4))
        trend = acc.build_trend()
        assert trend["aggregate"]["trend_direction"] == "decreasing"

    def test_build_trend_stable(self):
        acc = ReportAccumulator()
        acc.add_session(self._make_report(0.5))
        acc.add_session(self._make_report(0.5))
        trend = acc.build_trend()
        assert trend["aggregate"]["trend_direction"] == "stable"

    def test_empty_accumulator(self):
        acc = ReportAccumulator()
        trend = acc.build_trend()
        assert trend["sessions"] == []
        assert trend["aggregate"] == {}

    def test_export_trend_json(self):
        acc = ReportAccumulator()
        acc.add_session(self._make_report(0.4, flagged=1))
        json_str = acc.export_trend_json()
        parsed = json.loads(json_str)
        assert "sessions" in parsed
        assert "aggregate" in parsed

    def test_reset(self):
        acc = ReportAccumulator()
        acc.add_session(self._make_report(0.5))
        acc.reset()
        assert acc.session_count == 0

    def test_session_label(self):
        acc = ReportAccumulator()
        r = self._make_report(0.5)
        acc.add_session(r, session_label="Upload Batch #1")
        trend = acc.build_trend()
        assert trend["sessions"][0]["label"] == "Upload Batch #1"
