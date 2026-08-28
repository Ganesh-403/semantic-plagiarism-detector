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
Plagiarism Risk Scoring Engine
==============================
Comprehensive risk assessment engine that scores documents for plagiarism risk,
identifies high-risk patterns, and provides detailed breakdowns.

Features:
- Multi-dimensional risk scoring (lexical, semantic, structural, metadata)
- Risk pattern detection and classification
- Historical risk trend analysis
- Batch risk assessment with priority queuing
- Risk mitigation recommendations
"""

import hashlib
import json
import math
import random
import uuid
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class RiskDimension:
    """A single risk dimension score."""

    name: str
    score: float  # 0-100
    weight: float
    description: str
    indicators: List[str]
    severity: str  # critical, high, medium, low


@dataclass
class DocumentRisk:
    """Complete risk assessment for a document."""

    doc_id: str
    doc_title: str
    author: str
    upload_date: str
    overall_risk: float  # 0-100
    risk_level: str  # critical, high, medium, low
    dimensions: List[RiskDimension]
    matched_sources: List[MatchedSource]
    risk_factors: List[str]
    recommendations: List[str]
    scan_duration: float
    word_count: int
    language: str
    status: str  # scanned, pending, flagged, cleared


@dataclass
class MatchedSource:
    """A source matched against the document."""

    source_id: str
    source_title: str
    source_url: str
    similarity_score: float
    matched_sections: int
    total_sections: int
    overlap_percentage: float
    match_type: str  # exact, paraphrase, structural, semantic
    first_matched_position: int
    risk_contribution: float


@dataclass
class RiskPattern:
    """A detected plagiarism risk pattern."""

    pattern_id: str
    pattern_type: str  # copy_paste, paraphrase_heavy, structural_clone, source_hiding
    severity: str
    confidence: float
    affected_docs: List[str]
    evidence: Dict[str, Any]
    description: str
    mitigation: str
    detected_at: str


@dataclass
class RiskTrend:
    """Historical risk trend data point."""

    date: str
    avg_risk: float
    docs_scanned: int
    flagged_count: int
    critical_count: int
    top_risk_type: str


# =============================================================================
# MOCK DATA
# =============================================================================


def generate_documents(count: int = 25) -> List[DocumentRisk]:
    """Generate mock document risk assessments."""
    titles = [
        "Deep Learning Approaches for Natural Language Understanding",
        "A Survey of Transformer-Based Architectures",
        "Novel Methods for Semantic Text Similarity",
        "Code Clone Detection Using Machine Learning",
        "Automated Plagiarism Detection in Academic Writing",
        "Cross-Lingual Transfer Learning for Low-Resource Languages",
        "Knowledge Graph Construction from Unstructured Text",
        "Real-Time Document Similarity Scoring at Scale",
        "Adversarial Attacks on Text Classification Models",
        "Federated Learning for Privacy-Preserving NLP",
        "Graph Neural Networks for Citation Analysis",
        "Multi-Modal Learning for Document Understanding",
        "Efficient Fine-Tuning of Large Language Models",
        "Bias Detection in Automated Grading Systems",
        "Zero-Shot Learning for Named Entity Recognition",
        "Temporal Information Extraction from News Articles",
        "Sentiment Analysis Across Multiple Domains",
        "Automatic Summarization of Scientific Papers",
        "Question Answering Over Knowledge Bases",
        "Spam Detection in Academic Publishing",
        "Plagiarism Obfuscation Techniques and Countermeasures",
        "Semantic Web Technologies for Research Discovery",
        "Low-Resource Language Processing with Transfer Learning",
        "AI-Generated Text Detection Methods",
        "Benchmarking Embedding Models for Similarity Tasks",
    ]
    authors = [
        "Smith J.",
        "Chen W.",
        "Patel A.",
        "Kim S.",
        "Mueller K.",
        "Garcia M.",
        "Lee H.",
        "Brown T.",
        "Zhang Y.",
        "Johnson R.",
    ]
    languages = [
        "English",
        "English",
        "English",
        "French",
        "German",
        "English",
        "Spanish",
        "English",
    ]
    statuses = [
        "scanned",
        "scanned",
        "scanned",
        "flagged",
        "pending",
        "scanned",
        "cleared",
    ]

    docs = []
    for i in range(count):
        risk = random.uniform(5, 95)
        risk_level = (
            "critical"
            if risk > 80
            else "high"
            if risk > 60
            else "medium"
            if risk > 30
            else "low"
        )

        dimensions = [
            RiskDimension(
                "Lexical Overlap",
                random.uniform(5, 90),
                0.25,
                "Word-level similarity with known sources",
                ["Exact phrase matches", "N-gram overlap", "Stop word patterns"],
                "high" if random.random() > 0.5 else "medium",
            ),
            RiskDimension(
                "Semantic Similarity",
                random.uniform(10, 85),
                0.30,
                "Meaning-level similarity using embeddings",
                ["Embedding distance", "Paraphrase detection", "Concept overlap"],
                "high" if random.random() > 0.6 else "low",
            ),
            RiskDimension(
                "Structural Similarity",
                random.uniform(5, 70),
                0.20,
                "Document structure and organization patterns",
                ["Section ordering", "Paragraph structure", "Citation patterns"],
                "medium" if random.random() > 0.5 else "low",
            ),
            RiskDimension(
                "Metadata Analysis",
                random.uniform(0, 50),
                0.15,
                "Author, date, and source metadata consistency",
                ["Author history", "Date anomalies", "Source provenance"],
                "low" if random.random() > 0.3 else "medium",
            ),
            RiskDimension(
                "Source Quality",
                random.uniform(10, 60),
                0.10,
                "Quality and legitimacy of cited sources",
                ["DOI validity", "Journal reputation", "Citation density"],
                "high" if random.random() > 0.7 else "low",
            ),
        ]

        matched = []
        for j in range(random.randint(1, 5)):
            matched.append(
                MatchedSource(
                    source_id=f"src_{random.randint(1,100)}",
                    source_title=f"Source Paper {random.randint(1000,9999)}",
                    source_url=f"https://doi.org/10.{random.randint(1000,9999)}/{random.randint(10000,99999)}",
                    similarity_score=round(random.uniform(0.15, 0.95), 3),
                    matched_sections=random.randint(1, 8),
                    total_sections=random.randint(5, 15),
                    overlap_percentage=round(random.uniform(5, 60), 1),
                    match_type=random.choice(
                        ["exact", "paraphrase", "structural", "semantic"]
                    ),
                    first_matched_position=random.randint(1, 200),
                    risk_contribution=round(random.uniform(5, 25), 1),
                )
            )

        risk_factors = []
        if risk > 70:
            risk_factors.extend(
                [
                    "High lexical overlap with multiple sources",
                    "Suspicious citation patterns",
                ]
            )
        if risk > 50:
            risk_factors.extend(["Structural similarity detected", "Unusual metadata"])
        if risk > 30:
            risk_factors.append("Moderate semantic overlap")

        recommendations = []
        if risk > 80:
            recommendations.extend(
                [
                    "Immediate manual review required",
                    "Contact original authors",
                    "Check for duplicate publication",
                ]
            )
        elif risk > 60:
            recommendations.extend(
                ["Detailed similarity report needed", "Verify source attributions"]
            )
        elif risk > 30:
            recommendations.append("Review flagged sections for proper citation")

        docs.append(
            DocumentRisk(
                doc_id=f"doc_{i+1:03d}",
                doc_title=titles[i % len(titles)],
                author=random.choice(authors),
                upload_date=(
                    datetime.now() - timedelta(days=random.randint(0, 90))
                ).strftime("%Y-%m-%d"),
                overall_risk=round(risk, 1),
                risk_level=risk_level,
                dimensions=dimensions,
                matched_sources=matched,
                risk_factors=risk_factors,
                recommendations=recommendations,
                scan_duration=round(random.uniform(2, 45), 1),
                word_count=random.randint(2000, 15000),
                language=random.choice(languages),
                status=random.choice(statuses),
            )
        )

    return sorted(docs, key=lambda d: d.overall_risk, reverse=True)


def generate_patterns() -> List[RiskPattern]:
    """Generate plagiarism risk patterns."""
    patterns = [
        RiskPattern(
            pattern_id="PAT-001",
            pattern_type="copy_paste",
            severity="critical",
            confidence=0.92,
            affected_docs=["doc_001", "doc_003"],
            evidence={
                "exact_matches": 47,
                "total_paragraphs": 120,
                "consecutive_matches": 12,
            },
            description="Document contains 47 exact-match paragraphs from 2 sources, with 12 consecutive matching blocks.",
            mitigation="Immediate manual review. Verify proper quotation and attribution. Check for duplicate publication.",
            detected_at=(datetime.now() - timedelta(hours=3)).isoformat(),
        ),
        RiskPattern(
            pattern_id="PAT-002",
            pattern_type="paraphrase_heavy",
            severity="high",
            confidence=0.85,
            affected_docs=["doc_005", "doc_008", "doc_012"],
            evidence={
                "paraphrase_ratio": 0.68,
                "avg_edit_distance": 0.22,
                "synonym_substitution_rate": 0.45,
            },
            description="3 documents show heavy paraphrasing (68% of content) with high synonym substitution rates.",
            mitigation="Review paraphrased sections. Compare semantic meaning with sources. Check citation adequacy.",
            detected_at=(datetime.now() - timedelta(hours=8)).isoformat(),
        ),
        RiskPattern(
            pattern_id="PAT-003",
            pattern_type="structural_clone",
            severity="high",
            confidence=0.78,
            affected_docs=["doc_002", "doc_007"],
            evidence={
                "section_order_similarity": 0.91,
                "heading_match_rate": 0.85,
                "paragraph_count_ratio": 0.95,
            },
            description="2 documents share 91% structural similarity with matching section ordering and headings.",
            mitigation="Compare document structures. Verify independent research vs. derivative work.",
            detected_at=(datetime.now() - timedelta(days=1)).isoformat(),
        ),
        RiskPattern(
            pattern_id="PAT-004",
            pattern_type="source_hiding",
            severity="critical",
            confidence=0.88,
            affected_docs=["doc_004"],
            evidence={
                "hidden_citations": 8,
                "removed_metadata": True,
                "modified_dates": 3,
            },
            description="Document appears to have removed or modified source attributions and metadata.",
            mitigation="Restore original metadata. Cross-reference with known publication databases. Contact author.",
            detected_at=(datetime.now() - timedelta(hours=1)).isoformat(),
        ),
        RiskPattern(
            pattern_id="PAT-005",
            pattern_type="copy_paste",
            severity="medium",
            confidence=0.72,
            affected_docs=["doc_010", "doc_015", "doc_020"],
            evidence={
                "code_block_matches": 15,
                "api_pattern_matches": 8,
                "comment_similarities": 22,
            },
            description="Multiple code blocks and API usage patterns match known open-source projects without attribution.",
            mitigation="Add proper open-source license attribution. Verify compliance with original licenses.",
            detected_at=(datetime.now() - timedelta(days=2)).isoformat(),
        ),
    ]
    return patterns


def generate_trends(days: int = 30) -> List[RiskTrend]:
    """Generate historical risk trend data."""
    trends = []
    for i in range(days):
        date = (datetime.now() - timedelta(days=days - i)).strftime("%Y-%m-%d")
        avg_risk = random.uniform(25, 65)
        docs_scanned = random.randint(5, 30)
        flagged = int(docs_scanned * random.uniform(0.1, 0.3))
        critical = int(flagged * random.uniform(0.05, 0.15))
        trends.append(
            RiskTrend(
                date=date,
                avg_risk=round(avg_risk, 1),
                docs_scanned=docs_scanned,
                flagged_count=flagged,
                critical_count=critical,
                top_risk_type=random.choice(
                    ["lexical", "semantic", "structural", "metadata"]
                ),
            )
        )
    return trends


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def risk_color(risk: float) -> str:
    if risk > 80:
        return "#ef4444"
    if risk > 60:
        return "#f97316"
    if risk > 30:
        return "#eab308"
    return "#22c55e"


def severity_icon(sev: str) -> str:
    return {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(sev, "⚪")


def render_kpi(label: str, value: str, subtitle: str, color: str) -> None:
    st.markdown(
        f"""
    <div style="background:rgba(255,255,255,0.06);border-radius:14px;padding:18px 14px;
         border:1px solid rgba(255,255,255,0.08);text-align:center;">
        <div style="font-size:26px;font-weight:800;color:{color};margin-bottom:4px;">{value}</div>
        <div style="font-size:12px;font-weight:600;color:#e2e8f0;margin-bottom:2px;">{label}</div>
        <div style="font-size:10px;color:#94a3b8;">{subtitle}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_risk_gauge(risk: float, size: int = 120) -> str:
    """Render a risk gauge as HTML."""
    color = risk_color(risk)
    r = size // 2 - 8
    angle = (risk / 100) * 180
    return f"""
    <div style="text-align:center;">
        <svg width="{size}" height="{size//2 + 20}" viewBox="0 0 {size} {size//2 + 20}">
            <path d="M 8 {size//2} A {r} {r} 0 0 1 {size-8} {size//2}"
                  fill="none" stroke="rgba(255,255,255,0.08)" stroke-width="8" stroke-linecap="round"/>
            <path d="M 8 {size//2} A {r} {r} 0 0 1 {8 + (size-16) * math.sin(math.radians(angle)):,.0f} {size//2 - (size-16)/2 * math.cos(math.radians(angle)):,.0f}"
                  fill="none" stroke="{color}" stroke-width="8" stroke-linecap="round"/>
            <text x="{size//2}" y="{size//2 - 10}" text-anchor="middle" fill="#e2e8f0"
                  font-size="{size//5}" font-weight="800">{risk:.0f}</text>
            <text x="{size//2}" y="{size//2 + 8}" text-anchor="middle" fill="#94a3b8"
                  font-size="{size//12}">risk score</text>
        </svg>
    </div>
    """


def render_document_card(doc: DocumentRisk, expanded: bool = False) -> None:
    rc = risk_color(doc.overall_risk)
    status_colors = {
        "scanned": "#3b82f6",
        "pending": "#f59e0b",
        "flagged": "#ef4444",
        "cleared": "#22c55e",
    }
    st.markdown(
        f"""
    <div style="background:rgba(255,255,255,0.06);border-radius:12px;padding:14px;
         border-left:4px solid {rc};margin-bottom:10px;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
            <div>
                <span style="font-weight:700;font-size:14px;color:#e2e8f0;">{doc.doc_title}</span>
                <span style="font-size:11px;color:#94a3b8;margin-left:8px;">by {doc.author}</span>
            </div>
            <div style="display:flex;align-items:center;gap:8px;">
                <span style="font-size:10px;padding:2px 8px;border-radius:8px;background:{status_colors.get(doc.status, '#6b7280')}20;color:{status_colors.get(doc.status, '#6b7280')};text-transform:capitalize;font-weight:600;">{doc.status}</span>
                <span style="font-size:18px;font-weight:800;color:{rc};">{doc.overall_risk:.0f}</span>
            </div>
        </div>
        <div style="display:flex;gap:12px;font-size:11px;color:#94a3b8;margin-bottom:6px;">
            <span>📄 {doc.word_count:,} words</span>
            <span>🌍 {doc.language}</span>
            <span>📅 {doc.upload_date}</span>
            <span>🔍 {doc.scan_duration:.0f}s scan</span>
            <span>📎 {len(doc.matched_sources)} sources matched</span>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    if expanded:
        st.markdown("**Risk Dimensions:**")
        for dim in doc.dimensions:
            dim_color = risk_color(dim.score)
            st.markdown(
                f"""
            <div style="margin-bottom:8px;">
                <div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:3px;">
                    <span style="color:#e2e8f0;font-weight:600;">{dim.name}</span>
                    <span style="color:{dim_color};font-weight:700;">{dim.score:.0f}/100 (×{dim.weight})</span>
                </div>
                <div style="height:8px;background:rgba(255,255,255,0.08);border-radius:4px;">
                    <div style="height:100%;width:{dim.score}%;background:{dim_color};border-radius:4px;"></div>
                </div>
                <div style="font-size:10px;color:#94a3b8;margin-top:2px;">{dim.description}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        if doc.matched_sources:
            st.markdown("**Matched Sources:**")
            for src in doc.matched_sources:
                src_color = risk_color(src.similarity_score * 100)
                st.markdown(
                    f"""
                <div style="background:rgba(255,255,255,0.04);border-radius:8px;padding:10px;margin-bottom:6px;border-left:3px solid {src_color};">
                    <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                        <span style="font-size:12px;font-weight:600;color:#e2e8f0;">{src.source_title}</span>
                        <span style="font-size:11px;font-weight:700;color:{src_color};">{src.similarity_score:.0%} similarity</span>
                    </div>
                    <div style="font-size:11px;color:#94a3b8;">
                        📝 {src.matched_sections}/{src.total_sections} sections · 📊 {src.overlap_percentage}% overlap · 🔗 {src.match_type.replace('_', ' ').title()}
                    </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

        if doc.risk_factors:
            st.markdown("**Risk Factors:**")
            for rf in doc.risk_factors:
                st.markdown(
                    f'<div style="font-size:12px;color:#f97316;margin-bottom:2px;">⚠️ {rf}</div>',
                    unsafe_allow_html=True,
                )

        if doc.recommendations:
            st.markdown("**Recommendations:**")
            for rec in doc.recommendations:
                st.markdown(
                    f'<div style="font-size:12px;color:#22c55e;margin-bottom:2px;">✅ {rec}</div>',
                    unsafe_allow_html=True,
                )


def render_pattern_card(pattern: RiskPattern) -> None:
    sev = {
        "critical": "#ef4444",
        "high": "#f97316",
        "medium": "#eab308",
        "low": "#22c55e",
    }.get(pattern.severity, "#6b7280")
    st.markdown(
        f"""
    <div style="background:rgba(255,255,255,0.06);border-radius:12px;padding:14px;
         border-left:4px solid {sev};margin-bottom:10px;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
            <span style="font-weight:700;font-size:13px;color:#e2e8f0;">
                {pattern.pattern_id}: {pattern.pattern_type.replace('_', ' ').title()}
            </span>
            <div style="display:flex;gap:6px;">
                <span style="font-size:10px;padding:2px 8px;border-radius:8px;background:{sev}20;color:{sev};text-transform:uppercase;font-weight:600;">{pattern.severity}</span>
                <span style="font-size:10px;padding:2px 8px;border-radius:8px;background:rgba(255,255,255,0.06);color:#94a3b8;">{pattern.confidence:.0%} confidence</span>
            </div>
        </div>
        <div style="font-size:12px;color:#cbd5e1;margin-bottom:6px;">{pattern.description}</div>
        <div style="font-size:11px;color:#94a3b8;margin-bottom:4px;">📄 Affected: {', '.join(pattern.affected_docs)}</div>
        <div style="background:rgba(34,197,94,0.08);border-radius:8px;padding:8px;margin-top:6px;">
            <div style="font-size:11px;color:#22c55e;">💡 Mitigation: {pattern.mitigation}</div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


# =============================================================================
# MAIN DASHBOARD
# =============================================================================


def render_risk_scoring_engine() -> None:
    """Render the Plagiarism Risk Scoring Engine."""
    st.markdown(
        """
    <style>
    .block-container { padding-top: 1rem; }
    </style>
    """,
        unsafe_allow_html=True,
    )

    documents = generate_documents(25)
    patterns = generate_patterns()
    trends = generate_trends(30)

    # Header
    st.markdown(
        """
    <div style="text-align:center;margin-bottom:20px;">
        <div style="font-size:36px;margin-bottom:8px;">🛡️</div>
        <h1 style="font-size:28px;font-weight:800;margin:0;
            background:linear-gradient(135deg,#ef4444,#f59e0b);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            Plagiarism Risk Scoring Engine
        </h1>
        <p style="font-size:14px;color:#94a3b8;margin-top:6px;">
            Multi-dimensional risk assessment with pattern detection and mitigation
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # KPIs
    total_docs = len(documents)
    critical_docs = sum(1 for d in documents if d.risk_level == "critical")
    high_docs = sum(1 for d in documents if d.risk_level == "high")
    avg_risk = sum(d.overall_risk for d in documents) / len(documents)
    total_matches = sum(len(d.matched_sources) for d in documents)
    open_patterns = sum(1 for p in patterns if p.severity in ("critical", "high"))

    cols = st.columns(6)
    kpis = [
        ("Documents", str(total_docs), "scanned", "#3b82f6"),
        ("Critical Risk", str(critical_docs), "need immediate review", "#ef4444"),
        ("High Risk", str(high_docs), "require attention", "#f97316"),
        ("Avg Risk Score", f"{avg_risk:.0f}", "across all docs", risk_color(avg_risk)),
        ("Source Matches", str(total_matches), "across all docs", "#8b5cf6"),
        (
            "Active Patterns",
            str(open_patterns),
            "high/critical",
            "#ef4444" if open_patterns > 2 else "#22c55e",
        ),
    ]
    for col, (label, value, subtitle, color) in zip(cols, kpis):
        with col:
            render_kpi(label, value, subtitle, color)

    # Tabs
    tabs = ["📊 Overview", "📄 Documents", "🔍 Patterns", "📈 Trends"]
    selected = st.radio("Tabs", tabs, horizontal=True, label_visibility="collapsed")

    if selected == "📊 Overview":
        _render_overview(documents, patterns, trends)
    elif selected == "📄 Documents":
        _render_documents(documents)
    elif selected == "🔍 Patterns":
        _render_patterns(patterns, documents)
    elif selected == "📈 Trends":
        _render_trends(trends, documents)


def _render_overview(
    documents: List[DocumentRisk], patterns: List[RiskPattern], trends: List[RiskTrend]
) -> None:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📊 Risk Level Distribution")
        level_counts = Counter(d.risk_level for d in documents)
        for level in ["critical", "high", "medium", "low"]:
            count = level_counts.get(level, 0)
            pct = count / len(documents) if documents else 0
            col = risk_color(
                {"critical": 90, "high": 70, "medium": 45, "low": 10}[level]
            )
            st.markdown(
                f"""
            <div style="margin-bottom:8px;">
                <div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:3px;">
                    <span style="color:#e2e8f0;font-weight:600;text-transform:capitalize;">{level}</span>
                    <span style="color:#94a3b8;">{count} ({pct:.0%})</span>
                </div>
                <div style="height:10px;background:rgba(255,255,255,0.08);border-radius:5px;">
                    <div style="height:100%;width:{pct*100}%;background:{col};border-radius:5px;"></div>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        st.markdown("#### 🔍 Dimension Averages")
        dim_scores = defaultdict(list)
        for d in documents:
            for dim in d.dimensions:
                dim_scores[dim.name].append(dim.score)
        for name, scores in dim_scores.items():
            avg = sum(scores) / len(scores)
            st.markdown(
                f"""
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
                <span style="font-size:11px;color:#94a3b8;width:130px;">{name}</span>
                <div style="flex:1;height:8px;background:rgba(255,255,255,0.08);border-radius:4px;">
                    <div style="height:100%;width:{avg}%;background:{risk_color(avg)};border-radius:4px;"></div>
                </div>
                <span style="font-size:11px;color:{risk_color(avg)};font-weight:700;width:30px;">{avg:.0f}</span>
            </div>
            """,
                unsafe_allow_html=True,
            )

    with col2:
        st.markdown("#### 🕐 30-Day Risk Trend")
        if trends:
            max_docs = max(t.docs_scanned for t in trends)
            for t in trends[-15:]:
                bar_w = (t.docs_scanned / max_docs) * 100 if max_docs else 0
                st.markdown(
                    f"""
                <div style="display:flex;align-items:center;gap:8px;margin-bottom:3px;">
                    <span style="font-size:9px;color:#94a3b8;width:60px;">{t.date[5:]}</span>
                    <div style="flex:1;height:12px;background:rgba(255,255,255,0.08);border-radius:3px;position:relative;">
                        <div style="height:100%;width:{bar_w}%;background:{risk_color(t.avg_risk)};border-radius:3px;opacity:0.6;"></div>
                    </div>
                    <span style="font-size:10px;color:#94a3b8;width:24px;text-align:right;">{t.docs_scanned}</span>
                    <span style="font-size:10px;color:{risk_color(t.avg_risk)};font-weight:700;width:24px;">{t.avg_risk:.0f}</span>
                </div>
                """,
                    unsafe_allow_html=True,
                )

        st.markdown("#### 🔍 Active Risk Patterns")
        for p in patterns[:3]:
            render_pattern_card(p)


def _render_documents(documents: List[DocumentRisk]) -> None:
    col1, col2, col3 = st.columns(3)
    with col1:
        level_filter = st.selectbox(
            "Risk Level", ["All", "Critical", "High", "Medium", "Low"]
        )
    with col2:
        status_filter = st.selectbox(
            "Status", ["All", "Scanned", "Flagged", "Pending", "Cleared"]
        )
    with col3:
        sort_by = st.selectbox(
            "Sort By",
            ["Risk (High→Low)", "Risk (Low→High)", "Word Count", "Scan Duration"],
        )

    filtered = documents[:]
    if level_filter != "All":
        filtered = [d for d in filtered if d.risk_level == level_filter.lower()]
    if status_filter != "All":
        filtered = [d for d in filtered if d.status == status_filter.lower()]

    if sort_by == "Risk (High→Low)":
        filtered.sort(key=lambda d: d.overall_risk, reverse=True)
    elif sort_by == "Risk (Low→High)":
        filtered.sort(key=lambda d: d.overall_risk)
    elif sort_by == "Word Count":
        filtered.sort(key=lambda d: d.word_count, reverse=True)
    else:
        filtered.sort(key=lambda d: d.scan_duration, reverse=True)

    st.markdown(f"**{len(filtered)} documents** matching filters")

    for doc in filtered:
        with st.expander(
            f"{doc.doc_title} — Risk: {doc.overall_risk:.0f} ({doc.risk_level})",
            expanded=False,
        ):
            render_document_card(doc, expanded=True)


def _render_patterns(
    patterns: List[RiskPattern], documents: List[DocumentRisk]
) -> None:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("#### 🔍 Detected Risk Patterns")
        for p in patterns:
            render_pattern_card(p)
    with col2:
        st.markdown("#### 📊 Pattern Statistics")
        type_counts = Counter(p.pattern_type for p in patterns)
        for ptype, count in type_counts.most_common():
            st.markdown(f"• **{ptype.replace('_', ' ').title()}**: {count}")

        st.markdown("#### 📊 Severity Breakdown")
        sev_counts = Counter(p.severity for p in patterns)
        for sev, count in sev_counts.most_common():
            sev_col = {
                "critical": "#ef4444",
                "high": "#f97316",
                "medium": "#eab308",
                "low": "#22c55e",
            }[sev]
            st.markdown(
                f"""
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
                <span style="font-size:11px;color:#94a3b8;width:70px;">{sev.title()}</span>
                <div style="flex:1;height:10px;background:rgba(255,255,255,0.08);border-radius:5px;">
                    <div style="height:100%;width:{count/len(patterns)*100}%;background:{sev_col};border-radius:5px;"></div>
                </div>
                <span style="font-size:11px;color:{sev_col};font-weight:700;">{count}</span>
            </div>
            """,
                unsafe_allow_html=True,
            )

        st.markdown("#### 📊 Average Confidence")
        avg_conf = sum(p.confidence for p in patterns) / len(patterns)
        st.markdown(
            f'<div style="font-size:24px;font-weight:800;color:#f59e0b;text-align:center;">{avg_conf:.0%}</div>',
            unsafe_allow_html=True,
        )


def _render_trends(trends: List[RiskTrend], documents: List[DocumentRisk]) -> None:
    st.markdown("#### 📈 Risk Trend Analysis")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Documents Scanned Over Time**")
        max_d = max(t.docs_scanned for t in trends)
        for t in trends:
            bar_w = (t.docs_scanned / max_d) * 100 if max_d else 0
            st.markdown(
                f"""
            <div style="display:flex;align-items:center;gap:6px;margin-bottom:2px;">
                <span style="font-size:9px;color:#94a3b8;width:50px;">{t.date[5:]}</span>
                <div style="flex:1;height:10px;background:rgba(255,255,255,0.08);border-radius:3px;">
                    <div style="height:100%;width:{bar_w}%;background:#3b82f6;border-radius:3px;"></div>
                </div>
                <span style="font-size:10px;color:#94a3b8;width:20px;">{t.docs_scanned}</span>
            </div>
            """,
                unsafe_allow_html=True,
            )

    with col2:
        st.markdown("**Flagged Documents Over Time**")
        for t in trends:
            if t.flagged_count > 0:
                bar_w = (
                    (t.flagged_count / t.docs_scanned) * 100 if t.docs_scanned else 0
                )
                st.markdown(
                    f"""
                <div style="display:flex;align-items:center;gap:6px;margin-bottom:2px;">
                    <span style="font-size:9px;color:#94a3b8;width:50px;">{t.date[5:]}</span>
                    <div style="flex:1;height:10px;background:rgba(255,255,255,0.08);border-radius:3px;">
                        <div style="height:100%;width:{bar_w}%;background:#ef4444;border-radius:3px;"></div>
                    </div>
                    <span style="font-size:10px;color:#ef4444;width:20px;">{t.flagged_count}</span>
                </div>
                """,
                    unsafe_allow_html=True,
                )

    # Risk Distribution Summary
    st.markdown("#### 📊 Overall Risk Distribution Summary")
    all_dims = defaultdict(list)
    for d in documents:
        for dim in d.dimensions:
            all_dims[dim.name].append(dim.score)

    col_a, col_b = st.columns(2)
    with col_a:
        for name, scores in all_dims.items():
            avg = sum(scores) / len(scores)
            min_s = min(scores)
            max_s = max(scores)
            st.markdown(
                f"""
            <div style="background:rgba(255,255,255,0.04);border-radius:8px;padding:10px;margin-bottom:6px;">
                <div style="font-size:12px;font-weight:600;color:#e2e8f0;margin-bottom:4px;">{name}</div>
                <div style="display:flex;gap:8px;font-size:11px;color:#94a3b8;">
                    <span>Avg: <b style="color:{risk_color(avg)}">{avg:.1f}</b></span>
                    <span>Min: {min_s:.1f}</span>
                    <span>Max: {max_s:.1f}</span>
                    <span>Count: {len(scores)}</span>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    with col_b:
        st.markdown("**Top 10 Highest Risk Documents**")
        for i, doc in enumerate(documents[:10]):
            rc = risk_color(doc.overall_risk)
            st.markdown(
                f"""
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;padding:8px;background:rgba(255,255,255,0.04);border-radius:8px;">
                <span style="font-size:12px;font-weight:700;color:#94a3b8;width:20px;">#{i+1}</span>
                <span style="flex:1;font-size:12px;color:#e2e8f0;font-weight:600;">{doc.doc_title[:45]}...</span>
                <span style="font-size:14px;font-weight:800;color:{rc};">{doc.overall_risk:.0f}</span>
            </div>
            """,
                unsafe_allow_html=True,
            )


# =============================================================================
# ENTRY POINT
# =============================================================================


def main():
    render_risk_scoring_engine()


if __name__ == "__main__":
    main()
