"""Read actual artifacts produced from analysis results, including failure paths."""

import base64
from copy import deepcopy
import csv
from datetime import datetime, timedelta
from html.parser import HTMLParser
import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image
from pypdf import PdfReader
import pytest

from src.models.report import (
    Report,
    ReportConfig,
    ReportFormat,
    ReportRequest,
    ReportStatus,
    ReportType,
)
from src.reports import (
    CSVExporter,
    HTMLGenerator,
    PDFGenerator,
    ReportGenerator,
    ReportVisualizer,
)


@pytest.fixture
def analysis():
    return {
        "document_names": [
            "Ocean essay",
            "Weather essay",
            "Farming essay",
            "Space essay",
        ],
        "similarity_matrix": [
            [1, 0.85, 0.5, 0.3],
            [0.85, 1, 0.15, 0.1],
            [0.5, 0.15, 1, 0.95],
            [0.3, 0.1, 0.95, 1],
        ],
    }


@pytest.fixture
def generator(tmp_path):
    return ReportGenerator(str(tmp_path / "reports"))


class ParsedHTML(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.tags = []
        self.text = []
        self.images = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        if tag == "img":
            self.images.append(dict(attrs)["src"])

    def handle_data(self, data):
        self.text.append(data)


def assert_png(encoded):
    with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
        assert image.format == "PNG"
        assert image.width > 100 and image.height > 100
        assert image.convert("RGB").getextrema() != ((255, 255),) * 3
        image.load()


@pytest.mark.parametrize("format", list(ReportFormat))
def test_each_format_contains_supplied_results_and_completes(
    generator, analysis, format
):
    original = deepcopy(analysis)
    response = generator.generate_report(
        ReportRequest(
            "actual-analysis",
            analysis,
            format=format,
            title="Course report",
            description="September review",
        )
    )
    assert response.success, response.error
    report = response.report
    assert generator.get_report(response.report_id) is report
    assert report.status == ReportStatus.COMPLETED and report.progress == 100
    assert response.record_count == 4
    assert report.document_names == analysis["document_names"]
    assert analysis == original
    assert report.statistics["total_documents"] == 4
    assert report.statistics["total_comparisons"] == 6
    assert report.statistics["average_similarity"] == pytest.approx(0.475)
    assert report.statistics["high_severity_count"] == 2
    assert report.statistics["medium_severity_count"] == 1
    assert report.statistics["low_severity_count"] == 1
    path = Path(response.file_path)
    assert path.stat().st_size == response.file_size > 100
    assert path.suffix == "." + format.value
    assert list(path.parent.iterdir()) == [path]
    if format == ReportFormat.JSON:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["statistics"] == report.statistics
        assert payload["matches"] == report.matches
        assert payload["status"] == "completed" and payload["progress"] == 100
        assert "file_size" not in payload and "file_path" not in payload
    elif format == ReportFormat.CSV:
        rows = list(csv.reader(io.StringIO(path.read_text(encoding="utf-8"))))
        assert ["Average Similarity", "0.4750"] in rows
        assert ["Total Comparisons", "6"] in rows
        assert ["Ocean essay", "1.0000", "0.8500", "0.5000", "0.3000"] in rows
        assert ["1", "Ocean essay", "Weather essay", "", "", "0.8500", "High"] in rows
    elif format == ReportFormat.HTML:
        parsed = ParsedHTML(path.read_text(encoding="utf-8"))
        assert parsed.tags[:2] == ["html", "head"]
        assert "Course report" in parsed.text
        assert "47.50%" in parsed.text and "Ocean essay" in parsed.text
        assert any("Completed" in text for text in parsed.text)
        assert parsed.images
        for image in parsed.images:
            assert_png(image.split(",", 1)[1])
    else:
        pdf = PdfReader(io.BytesIO(path.read_bytes()))
        assert len(pdf.pages) >= 4
        text = "\n".join(page.extract_text() for page in pdf.pages)
        assert "Course report" in text and "Ocean essay" in text
        assert "47.50%" in text and "85.00%" in text
        assert any(page.images for page in pdf.pages)
    stats = generator.get_generation_stats()
    assert stats["total_generated"] == stats["successful"] == 1
    assert stats["failed"] == 0 and stats["success_rate"] == 100
    assert stats["average_time_ms"] > 0


@pytest.mark.parametrize("format", list(ReportFormat))
def test_config_applies_to_exported_content(generator, analysis, format):
    response = generator.generate_report(
        ReportRequest(
            "a",
            analysis,
            format=format,
            config=ReportConfig(
                include_heatmap=False, include_summary_stats=False, max_matches=1
            ),
        )
    )
    assert response.success, response.error
    assert response.record_count == 1
    assert len(response.report.matches) == 4  # Registry retains the complete analysis.
    data = Path(response.file_path).read_bytes()
    if format == ReportFormat.JSON:
        content = json.loads(data)
        assert len(content["matches"]) == 1
        assert content["statistics"] == {} and content["similarity_matrix"] == []
    elif format == ReportFormat.PDF:
        pdf = PdfReader(io.BytesIO(data))
        text = "\n".join(page.extract_text() for page in pdf.pages)
        assert "85.00%" in text and "95.00%" not in text
        assert "Similarity Heatmap" not in text and "Executive Summary" not in text
    else:
        text = data.decode("utf-8")
        assert "95.00%" not in text and "0.9500" not in text
        assert "Executive Summary" not in text and "SIMILARITY MATRIX" not in text


def test_summary_only_and_empty_analysis_do_not_invent_matches(generator, analysis):
    config = ReportConfig()
    response = generator.generate_report(
        ReportRequest(
            "a", analysis, format="json", include_details=False, config=config
        )
    )
    payload = json.loads(Path(response.file_path).read_text())
    assert payload["matches"] == payload["similarity_matrix"] == []
    assert payload["statistics"]["total_documents"] == 4
    assert config.include_heatmap and config.include_matches
    for names, matrix in (([], []), (["Only document"], [[1]])):
        response = generator.generate_report(
            ReportRequest(
                "empty",
                {"document_names": names, "similarity_matrix": matrix},
                format="json",
                config=None,
            )
        )
        assert response.success
        assert response.record_count == 0
        assert response.report.statistics["average_similarity"] == 0
        assert response.report.statistics["total_comparisons"] == 0


@pytest.mark.parametrize(
    "changes",
    [
        {"document_names": ["duplicate", "duplicate"]},
        {"document_names": "not a list"},
        {"document_names": [""]},
        {"similarity_matrix": [[1, 0.5]]},
        {"similarity_matrix": [[1, float("nan")], [float("nan"), 1]]},
        {"similarity_matrix": [[1, 2], [2, 1]]},
        {"similarity_matrix": [[1, 0.5], [0.6, 1]]},
        {"matches": "invalid"},
        {"matches": [123]},
        {"matches": [{"source_document": "a", "target_document": "a", "score": 0.4}]},
        {
            "matches": [
                {"source_document": "missing", "target_document": "b", "score": 0.4}
            ]
        },
        {"matches": [{"source_document": "a", "target_document": "b"}]},
        {
            "matches": [
                {"source_document": "a", "target_document": "b", "score": float("inf")}
            ]
        },
        {
            "matches": [
                {
                    "source_document": "a",
                    "target_document": "b",
                    "score": 0.4,
                    "lexical_score": -1,
                }
            ]
        },
    ],
)
def test_bad_analysis_is_failed_and_does_not_publish(generator, changes):
    data = {
        "document_names": ["a", "b"],
        "similarity_matrix": [[1, 0.5], [0.5, 1]],
        **changes,
    }
    response = generator.generate_report(
        ReportRequest("analysis-id", data, format="json")
    )
    assert not response.success and response.error
    assert response.report.status == ReportStatus.FAILED
    assert response.report.id != "analysis-id"
    assert generator.get_report(response.report_id).error == response.error
    assert list(generator.output_dir.iterdir()) == []
    assert generator.get_generation_stats()["failed"] == 1


@pytest.mark.parametrize("data", [{}, None, [], {"document_names": ["a"]}])
def test_results_must_be_supplied(generator, data):
    response = generator.generate_report(ReportRequest("missing", data, format="json"))
    assert not response.success and "analysis_data" in response.error


def test_explicit_matches_are_validated_copied_and_not_replaced(generator, analysis):
    analysis["matches"] = [
        {
            "source_document": "Ocean essay",
            "target_document": "Weather essay",
            "score": 0.1,
            "lexical_score": 0.05,
            "semantic_score": 0.15,
        }
    ]
    result = generator.generate_report(ReportRequest("a", analysis, format="json"))
    assert result.success
    assert result.report.matches[0]["severity"] == "none"
    assert "severity" not in analysis["matches"][0]
    analysis["matches"][0]["score"] = 0.99
    assert result.report.matches[0]["score"] == 0.1


def test_atomic_publication_failure_preserves_existing_file_and_tracks_failure(
    generator, analysis, monkeypatch
):
    import src.reports.report_generator as module

    target = generator.output_dir / "stable.json"
    target.write_text("prior report")
    original_create = generator._create_report_from_request

    def create(request):
        report = original_create(request)
        del generator._reports[report.id]
        report.id = "stable"
        generator._reports[report.id] = report
        return report

    def fail_replace(source, destination):
        assert Path(source).read_bytes().startswith(b"{")
        raise OSError("disk publication failed")

    monkeypatch.setattr(generator, "_create_report_from_request", create)
    monkeypatch.setattr(module.os, "replace", fail_replace)
    result = generator.generate_report(ReportRequest("a", analysis, format="json"))
    assert not result.success and result.report.status == ReportStatus.FAILED
    assert target.read_text() == "prior report"
    assert list(generator.output_dir.iterdir()) == [target]


def test_registry_pagination_cleanup_and_path_guards(
    generator, analysis, tmp_path, monkeypatch
):
    assert generator.get_generation_stats()["success_rate"] == 0
    reports = [
        generator.generate_report(ReportRequest(str(i), analysis, format="json")).report
        for i in range(3)
    ]
    reports[0].generated_at = datetime.now() - timedelta(days=40)
    assert generator.list_reports(1, 1) == [reports[1]]
    assert generator.list_reports(0) == []
    assert generator.get_report("missing") is None and not generator.delete_report(
        "missing"
    )
    for kwargs in ({"limit": -1}, {"offset": -1}):
        with pytest.raises(ValueError):
            generator.list_reports(**kwargs)
    with pytest.raises(ValueError):
        generator.cleanup_old_reports(-1)
    assert generator.cleanup_old_reports(30) == 1
    assert not Path(reports[0].file_path).exists()
    outside = tmp_path / "outside.json"
    outside.write_text("preserve")
    original_path = reports[1].file_path
    reports[1].file_path = str(outside)
    with pytest.raises(ValueError):
        generator.delete_report(reports[1].id)
    assert outside.read_text() == "preserve"
    reports[1].file_path = original_path
    Path(original_path).unlink()
    assert generator.delete_report(reports[1].id)
    report = reports[2]
    real_unlink = Path.unlink

    def denied(path, *args, **kwargs):
        if path == Path(report.file_path):
            raise PermissionError("in use")
        return real_unlink(path, *args, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "unlink", denied)
        with pytest.raises(PermissionError):
            generator.delete_report(report.id)
    assert generator.get_report(report.id) is report
    assert generator.delete_report(report.id)
    report.id = "../escaped"
    with pytest.raises(ValueError, match="inside"):
        generator._export_report(
            report, {"matches": [], "summary": {}, "similarity_matrix": []}, "json"
        )
    assert not (tmp_path / "escaped.json").exists()


def test_renderers_escape_untrusted_text_and_share_severity_boundaries():
    report = Report(title="<script>alert(1)</script>", description="A & B <unclosed>")
    report.document_names = ["=1+1", "\t@sum(1)"]
    report.similarity_matrix = [[1, 0.5], [0.5, 1]]
    report.matches = [
        {
            "source_document": "<img src=x onerror=alert(1)>",
            "target_document": "=1+1",
            "score": score,
        }
        for score in (0.8, 0.5, 0.3, 0.1)
    ]
    html = HTMLGenerator().generate(report, {})
    parsed = ParsedHTML(html)
    assert "script" not in parsed.tags and parsed.tags.count("img") == 1
    assert "<script>alert(1)</script>" in parsed.text
    assert "<img src=x onerror=alert(1)>" in parsed.text
    for severity in ("High", "Medium", "Low", "None"):
        assert severity in parsed.text
    pdf = PDFGenerator().generate(report, {})
    document = PdfReader(io.BytesIO(pdf))
    assert "<script>alert(1)</script>" in document.pages[0].extract_text()
    assert "A & B <unclosed>" in document.pages[0].extract_text()
    exporter = CSVExporter()
    rows = list(csv.reader(io.StringIO(exporter.export_matches_only(report))))
    assert [row[-1] for row in rows[1:]] == ["High", "Medium", "Low", "None"]
    assert all(row[1] == "'=1+1" for row in rows[1:])
    matrix = list(csv.reader(io.StringIO(exporter.export_matrix_only(report))))
    assert matrix[0] == ["", "'=1+1", "'\t@sum(1)"]
    assert matrix[1][0] == "'=1+1"
    assert exporter.export_to_dict(report)["matches"] == report.matches
    assert list(csv.reader(io.StringIO(exporter.export_summary_only(report))))[0] == [
        "Metric",
        "Value",
    ]
    assert "'=1+1" in exporter.export(report, {})
    exporter.delimiter = ";"
    assert "Source Document;Target Document" in exporter.export_matches_only(report)


def test_empty_renderers_and_report_lifecycle():
    report = Report("Empty", report_type=ReportType.SUMMARY)
    assert report.to_dict()["status"] == "pending"
    report.update_progress(45)
    for value in (-1, 101):
        with pytest.raises(ValueError):
            report.update_progress(value)
    report.mark_failed("first failure")
    report.mark_completed("complete.json", 42)
    assert report.error is None and report.progress == 100
    assert json.loads(json.dumps(report.to_dict()))["file_size"] == 42
    html = HTMLGenerator()
    assert html._generate_heatmap(report) == html._generate_matches(report) == ""
    assert "Empty" in html.generate(report, {})
    pdf = PDFGenerator()
    assert pdf._create_matches_table(report) is None
    assert pdf.generate(report, {}).startswith(b"%PDF")
    csv_exporter = CSVExporter()
    assert csv_exporter.export_matrix_only(report) == "No similarity matrix available"
    assert (
        len(list(csv.reader(io.StringIO(csv_exporter.export_matches_only(report)))))
        == 1
    )
    assert "DOCUMENTS ANALYZED" not in csv_exporter.export(report, {})


def test_pdf_tables_paginate_and_invalid_image_has_visible_fallback():
    report = Report("Long report")
    report.matches = [
        {
            "source_document": f"Source {i}",
            "target_document": f"Target {i}",
            "score": 0.85,
        }
        for i in range(150)
    ]
    report.document_names = ["a", "b"]
    report.similarity_matrix = [[1, 0.5], [0.5, 1]]
    payload = PDFGenerator().generate(
        report, {"visualizations": {"heatmap": "invalid image"}}
    )
    pdf = PdfReader(io.BytesIO(payload))
    text = "\n".join(page.extract_text() for page in pdf.pages)
    assert len(pdf.pages) > 3
    assert "Heatmap generation failed." in text
    assert "Source 0" in text and "Source 149" in text


def test_all_optional_sections_can_be_disabled(generator, analysis):
    result = generator.generate_report(
        ReportRequest(
            "a",
            analysis,
            config=ReportConfig(False, False, False, 0),
            format="html",
        )
    )
    assert result.success and result.record_count == 0
    parsed = ParsedHTML(Path(result.file_path).read_text(encoding="utf-8"))
    assert parsed.images == []
    assert "Detected Matches" not in " ".join(parsed.text)
    assert "Executive Summary" not in " ".join(parsed.text)
    assert ReportConfig(max_matches=0).to_dict()["max_matches"] == 0
    with pytest.raises(ValueError):
        ReportRequest("a", analysis, format="exe")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_matches": -1},
        {"max_matches": 1.5},
        {"highlight_threshold": -1},
        {"highlight_threshold": float("nan")},
    ],
)
def test_config_rejects_invalid_limits(kwargs):
    with pytest.raises(ValueError):
        ReportConfig(**kwargs)


def test_visualizations_produce_decodable_images_and_close_on_failure(monkeypatch):
    visualizer = ReportVisualizer()
    before = set(plt.get_fignums())
    for encoded in (
        visualizer.create_heatmap([[1, 0.2], [0.2, 1]], ["a", "b"], annotate=False),
        visualizer.create_similarity_chart([0.9, 0.4, 0.1], ["high", "medium", "low"]),
        visualizer.create_distribution_chart([0.9, 0.4, 0.1]),
        visualizer.create_severity_distribution(
            [{"score": s} for s in (0.9, 0.5, 0.3, 0.1)]
        ),
        visualizer.create_summary_dashboard({"average_similarity": 0.5}, []),
    ):
        assert_png(encoded)
    assert set(plt.get_fignums()) == before
    assert visualizer.create_heatmap([], []) == ""
    assert visualizer.create_similarity_chart([], []) == ""
    assert visualizer.create_distribution_chart([]) == ""
    assert visualizer.create_severity_distribution([]) == ""
    fig = plt.figure()

    def fail(*args, **kwargs):
        raise OSError("cannot encode image")

    monkeypatch.setattr(fig, "savefig", fail)
    with pytest.raises(OSError):
        visualizer._fig_to_base64(fig)
    assert set(plt.get_fignums()) == before


@pytest.mark.parametrize(
    "stats, expected",
    [
        ({"high_severity_count": 2, "total_matches": 3}, "More than half"),
        ({"high_severity_count": 1, "total_matches": 4}, "priority review"),
        ({"high_severity_count": 1, "total_matches": 0}, "priority review"),
        ({"average_similarity": 0.4}, "Moderate"),
        ({}, "Low overall"),
    ],
)
def test_insights_handle_statistics_consistently(generator, stats, expected):
    assert expected in generator._generate_insight(stats)
