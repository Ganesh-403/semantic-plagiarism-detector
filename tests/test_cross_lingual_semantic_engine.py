"""
Unit tests for Enterprise Cross-Lingual & Multi-Granularity Semantic Plagiarism Engine
"""

import pytest
from src.cross_lingual_semantic_engine import (
    CrossLingualSemanticAnalyzer,
    CodeASTStructureHasher,
    MultiGranularityParagraphAligner,
)
from src.plagiarism_report_exporter import PlagiarismReportExporter


def test_cross_lingual_semantic_indexing_and_similarity():
    analyzer = CrossLingualSemanticAnalyzer(embedding_dimension=256)
    analyzer.index_reference_document(
        "REF-001",
        "Deep neural networks use multi-layer perceptrons and backpropagation for gradient optimization.",
        metadata={"category": "ai_research"},
    )

    query = (
        "Deep neural networks utilize backpropagation algorithms for multi-layer"
        " gradient descent."
    )
    matches = analyzer.detect_cross_lingual_similarity(query, similarity_threshold=0.5)

    assert len(matches) > 0
    assert matches[0]["matched_doc_id"] == "REF-001"
    assert matches[0]["similarity_score"] >= 0.5
    assert matches[0]["metadata"]["category"] == "ai_research"


def test_code_ast_structure_hasher():
    code_a = """
def calculate_area(radius):
    if radius <= 0:
        return 0
    return 3.14159 * radius * radius
"""
    code_b = """
def compute_circle_area(r_val):
    if r_val <= 0:
        return 0
    return 3.14 * r_val * r_val
"""
    similarity = CodeASTStructureHasher.compute_structural_similarity(code_a, code_b)
    assert similarity >= 0.7


def test_paragraph_alignment_engine():
    aligner = MultiGranularityParagraphAligner()
    doc_q = (
        "Artificial intelligence and deep learning models have reshaped modern"
        " natural language processing.\n\nTransformers use attention mechanisms to"
        " capture contextual dependencies efficiently."
    )
    doc_r = (
        "Modern natural language processing has been reshaped by artificial"
        " intelligence and deep learning models.\n\nAttention mechanisms allow"
        " transformers to efficiently learn contextual dependencies."
    )

    matches = aligner.align_and_score_paragraphs(doc_q, doc_r)
    assert len(matches) > 0
    assert matches[0]["paragraph_similarity_score"] >= 0.70


def test_plagiarism_report_exporter():
    exporter = PlagiarismReportExporter(
        "Neural Network Optimization Thesis", "Dr. Alex Vance"
    )
    matched_sources = [
        {
            "matched_doc_id": "REF-001",
            "similarity_score": 0.92,
            "confidence_grade": "HIGH",
            "snippet": "Deep neural networks use multi-layer perceptrons...",
        }
    ]
    paragraph_matches = [
        {
            "query_paragraph_index": 0,
            "reference_paragraph_index": 0,
            "paragraph_similarity_score": 0.88,
            "query_snippet": "Artificial intelligence and deep learning...",
            "reference_snippet": "Modern natural language processing...",
        }
    ]

    md_report = exporter.generate_markdown_report(
        28.5,
        matched_sources,
        code_similarity_pct=85.0,
        paragraph_alignment_matches=paragraph_matches,
    )
    assert "FAILED" in md_report
    assert "Dr. Alex Vance" in md_report
    assert "Paragraph Level Paraphrase Alignments" in md_report

    html_report = exporter.export_html_report(28.5, matched_sources)
    assert "<h1>Plagiarism Analysis Audit</h1>" in html_report

    json_report = exporter.export_json_summary(28.5, matched_sources)
    assert json_report["status"] == "FLAGGED"
    assert json_report["matchesCount"] == 1
