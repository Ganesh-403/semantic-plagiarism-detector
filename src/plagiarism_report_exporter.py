# MIT License
#
# Copyright (c) 2026 Ganesh Kambli
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Enterprise Plagiarism Report Generator & Visual Telemetry Suite
Generates structured Markdown and HTML plagiarism analysis reports detailing text overlap,
paraphrase probability, source attribution, and document integrity telemetry.
"""

from typing import Any, Dict, List, Optional


class PlagiarismReportExporter:
    """
    Generates structured compliance and plagiarism audit reports for academic institutions.
    """

    def __init__(
        self,
        document_title: str,
        author_name: str,
        institution_id: Optional[str] = "CAMPUS-UNIV-01",
    ):
        self.document_title = document_title
        self.author_name = author_name
        self.institution_id = institution_id

    def generate_markdown_report(
        self,
        overall_similarity_pct: float,
        matched_sources: list[dict[str, Any]],
        code_similarity_pct: float = 0.0,
        paragraph_alignment_matches: Optional[list[dict[str, Any]]] = None,
    ) -> str:
        """Generates comprehensive Markdown report."""
        report = f"""# Plagiarism Analysis Report: {self.document_title}

## 📋 Document Overview & Telemetry
- **Author / Submitter:** {self.author_name}
- **Institution ID:** {self.institution_id}
- **Overall Text Plagiarism Risk:** {overall_similarity_pct}%
- **Code AST Structural Similarity:** {code_similarity_pct}%
- **Audit Compliance Status:** {"FAILED" if overall_similarity_pct > 25.0 else "PASSED"}

## 🔍 Top Matched Source References
"""
        for src in matched_sources:
            report += f"""
### Source ID: `{src.get('matched_doc_id', 'UNKNOWN')}`
- **Similarity Score:** {src.get('similarity_score', 0.0) * 100}%
- **Confidence Grade:** {src.get('confidence_grade', 'N/A')}
- **Matching Snippet:**
  > "{src.get('snippet', '')}"
"""

        if paragraph_alignment_matches:
            report += "\n## 🧩 Paragraph Level Paraphrase Alignments\n"
            for match in paragraph_alignment_matches[:5]:
                report += f"""
- **Query Paragraph #{match.get('query_paragraph_index')} vs Ref Paragraph #{match.get('reference_paragraph_index')}:**
  - Score: {match.get('paragraph_similarity_score') * 100}%
  - Snippet Query: "{match.get('query_snippet')}"
  - Snippet Ref: "{match.get('reference_snippet')}"
"""

        report += "\n---\n*Generated automatically by Semantic Plagiarism Detector Engine (Enterprise Suite)*"
        return report

    def export_html_report(
        self, overall_similarity_pct: float, matched_sources: list[dict[str, Any]]
    ) -> str:
        """Generates standalone HTML report with responsive CSS telemetry tables."""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Plagiarism Report - {self.document_title}</title>
    <style>
        body {{ font-family: sans-serif; background: #0f172a; color: #f8fafc; padding: 20px; }}
        .card {{ background: #1e293b; padding: 20px; border-radius: 12px; margin-bottom: 20px; }}
        .badge-fail {{ color: #ef4444; font-weight: bold; }}
        .badge-pass {{ color: #22c55e; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="card">
        <h1>Plagiarism Analysis Audit</h1>
        <p>Document: {self.document_title} | Author: {self.author_name}</p>
        <p>Overall Plagiarism Score: <strong>{overall_similarity_pct}%</strong></p>
        <p>Status: <span class="{"badge-fail" if overall_similarity_pct > 25.0 else "badge-pass"}">
            {"FLAGGED" if overall_similarity_pct > 25.0 else "CLEAN"}
        </span></p>
    </div>
</body>
</html>"""
        return html

    def export_json_summary(
        self, overall_similarity_pct: float, matched_sources: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Returns JSON report summary dict."""
        return {
            "documentTitle": self.document_title,
            "authorName": self.author_name,
            "institutionId": self.institution_id,
            "overallSimilarityPct": overall_similarity_pct,
            "status": "FLAGGED" if overall_similarity_pct > 20.0 else "CLEAN",
            "matchesCount": len(matched_sources),
            "matchedSources": matched_sources,
        }
