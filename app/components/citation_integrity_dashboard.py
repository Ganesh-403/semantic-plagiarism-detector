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
Citation Integrity Dashboard Component
=======================================
Monitors citation patterns, detects citation manipulation,
verifies source authenticity, and tracks citation network health.

Features:
- Citation pattern analysis and anomaly detection
- Source authenticity verification
- Citation network visualization
- Plagiarism-aware citation tracking
- Citation impact scoring
"""

import hashlib
import json
import math
import random
import time
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
class Citation:
    """Represents a single citation record."""

    id: str
    source_title: str
    source_authors: List[str]
    source_doi: Optional[str]
    source_url: str
    source_year: int
    source_journal: str
    cited_by_doc_id: str
    cited_by_doc_title: str
    citation_context: str
    citation_type: str  # direct_quote, paraphrase, reference, data
    confidence_score: float
    verified: bool
    flagged: bool
    created_at: str
    similarity_to_source: float
    context_relevance: float


@dataclass
class SourceAuthenticity:
    """Authenticity verification result for a cited source."""

    source_id: str
    title: str
    doi_valid: bool
    url_accessible: bool
    metadata_match: bool
    peer_reviewed: bool
    retracted: bool
    predatory_journal: bool
    authenticity_score: float
    risk_factors: List[str]
    verification_date: str


@dataclass
class CitationNetwork:
    """A node in the citation network graph."""

    doc_id: str
    doc_title: str
    citation_count: int
    cited_by_count: int
    self_citation_rate: float
    avg_source_quality: float
    top_journals: List[str]
    citation_diversity: float


@dataclass
class CitationAnomaly:
    """An anomalous citation pattern detected."""

    anomaly_id: str
    anomaly_type: (
        str  # self_citation_burst, citation_ring, fake_source, temporal_cluster
    )
    severity: str  # critical, high, medium, low
    description: str
    affected_docs: List[str]
    evidence: Dict[str, Any]
    detected_at: str
    status: str  # open, investigating, resolved, dismissed


@dataclass
class CitationImpact:
    """Impact metrics for a citation."""

    citation_id: str
    source_h_index: int
    source_citation_count: int
    source_age: int
    journal_impact_factor: float
    citation_velocity: float
    recency_weight: float
    impact_score: float


# =============================================================================
# MOCK DATA GENERATORS
# =============================================================================


def generate_citations(count: int = 30) -> List[Citation]:
    """Generate mock citation data."""
    journals = [
        "Nature",
        "Science",
        "IEEE Transactions",
        "ACM Computing Surveys",
        "Journal of Machine Learning Research",
        "Physical Review Letters",
        "The Lancet",
        "Cell",
        "PNAS",
        "Nature Communications",
        "arXiv preprint",
        "NeurIPS Proceedings",
        "ICML Proceedings",
    ]
    citation_types = ["direct_quote", "paraphrase", "reference", "data"]
    contexts = [
        "Building on the methodology described by {authors}, we...",
        "As demonstrated in {title}, the approach of...",
        "Following the framework established by {authors}...",
        "Recent work by {authors} shows that...",
        "In contrast to the findings of {title}, our results suggest...",
        "The dataset described in {title} was used to validate...",
        "Extending the analysis of {authors}, we observe...",
        "Similar to the approach in {title}, we employ...",
    ]
    doc_titles = [
        "Deep Learning for NLP",
        "Transformer Architecture Analysis",
        "Semantic Similarity Methods",
        "Plagiarism Detection Survey",
        "Code Clone Detection",
        "AI-Generated Text Identification",
        "Cross-lingual Transfer Learning",
        "Knowledge Graph Completion",
        "Information Retrieval Methods",
        "Text Summarization Techniques",
    ]

    citations = []
    for i in range(count):
        authors = [f"Author{random.randint(1,50)}" for _ in range(random.randint(1, 5))]
        title = f"Research Paper {random.randint(1000, 9999)}"
        doc_idx = random.randint(0, len(doc_titles) - 1)
        ctx = random.choice(contexts).format(
            authors=", ".join(authors[:2]),
            title=title,
        )
        conf = random.uniform(0.3, 1.0)
        citations.append(
            Citation(
                id=str(uuid.uuid4())[:8],
                source_title=title,
                source_authors=authors,
                source_doi=(
                    f"10.{random.randint(1000, 9999)}/{random.randint(10000, 99999)}"
                    if random.random() > 0.2
                    else None
                ),
                source_url=f"https://doi.org/10.{random.randint(1000, 9999)}/{random.randint(10000, 99999)}",
                source_year=random.randint(2015, 2026),
                source_journal=random.choice(journals),
                cited_by_doc_id=f"doc_{random.randint(1, 10)}",
                cited_by_doc_title=doc_titles[doc_idx],
                citation_context=ctx,
                citation_type=random.choice(citation_types),
                confidence_score=round(conf, 3),
                verified=random.random() > 0.3,
                flagged=random.random() > 0.85,
                created_at=(
                    datetime.now() - timedelta(days=random.randint(0, 365))
                ).isoformat(),
                similarity_to_source=round(random.uniform(0.2, 0.98), 3),
                context_relevance=round(random.uniform(0.3, 1.0), 3),
            )
        )
    return citations


def generate_authenticity(citations: List[Citation]) -> List[SourceAuthenticity]:
    """Generate authenticity verification results."""
    results = []
    seen = set()
    for c in citations:
        if c.source_title in seen:
            continue
        seen.add(c.source_title)
        doi_valid = c.source_doi is not None and random.random() > 0.1
        risk_factors = []
        if not doi_valid:
            risk_factors.append("No valid DOI")
        if random.random() > 0.9:
            risk_factors.append("Retracted paper")
        if random.random() > 0.85:
            risk_factors.append("Predatory journal")
        if random.random() > 0.8:
            risk_factors.append("URL inaccessible")

        score = 1.0
        for _ in risk_factors:
            score *= random.uniform(0.5, 0.8)

        results.append(
            SourceAuthenticity(
                source_id=str(uuid.uuid4())[:8],
                title=c.source_title,
                doi_valid=doi_valid,
                url_accessible=random.random() > 0.15,
                metadata_match=random.random() > 0.2,
                peer_reviewed=random.random() > 0.3,
                retracted="Retracted paper" in risk_factors,
                predatory_journal="Predatory journal" in risk_factors,
                authenticity_score=round(score, 3),
                risk_factors=risk_factors,
                verification_date=datetime.now().isoformat(),
            )
        )
    return results


def generate_anomalies() -> List[CitationAnomaly]:
    """Generate citation anomaly data."""
    anomalies = [
        CitationAnomaly(
            anomaly_id="ANOM-001",
            anomaly_type="self_citation_burst",
            severity="high",
            description="Document 'Deep Learning for NLP' has 45% self-citation rate, exceeding the 20% threshold.",
            affected_docs=["doc_1"],
            evidence={"self_cite_rate": 0.45, "threshold": 0.20},
            detected_at=(datetime.now() - timedelta(days=2)).isoformat(),
            status="investigating",
        ),
        CitationAnomaly(
            anomaly_id="ANOM-002",
            anomaly_type="citation_ring",
            severity="critical",
            description="Cluster of 5 documents citing each other in a circular pattern, suggesting coordinated citation manipulation.",
            affected_docs=["doc_3", "doc_4", "doc_5", "doc_6", "doc_7"],
            evidence={"ring_size": 5, "mutual_citations": 12},
            detected_at=(datetime.now() - timedelta(days=1)).isoformat(),
            status="open",
        ),
        CitationAnomaly(
            anomaly_id="ANOM-003",
            anomaly_type="fake_source",
            severity="critical",
            description="DOI 10.99999/99999 resolves to a non-existent paper. Likely fabricated citation.",
            affected_docs=["doc_2"],
            evidence={"doi": "10.99999/99999", "http_status": 404},
            detected_at=(datetime.now() - timedelta(hours=6)).isoformat(),
            status="open",
        ),
        CitationAnomaly(
            anomaly_id="ANOM-004",
            anomaly_type="temporal_cluster",
            severity="medium",
            description="8 citations added within 10 minutes on the same document, unusual pattern for manual citation.",
            affected_docs=["doc_8"],
            evidence={"citations_in_window": 8, "window_minutes": 10},
            detected_at=(datetime.now() - timedelta(hours=12)).isoformat(),
            status="resolved",
        ),
        CitationAnomaly(
            anomaly_id="ANOM-005",
            anomaly_type="self_citation_burst",
            severity="medium",
            description="Author 'Smith J.' appears in both citing and cited author lists 15 times across 3 documents.",
            affected_docs=["doc_9", "doc_10", "doc_11"],
            evidence={"author": "Smith J.", "appearances": 15},
            detected_at=(datetime.now() - timedelta(days=5)).isoformat(),
            status="dismissed",
        ),
    ]
    return anomalies


def generate_network() -> List[CitationNetwork]:
    """Generate citation network data."""
    docs = [
        ("doc_1", "Deep Learning for NLP", 24, 18, 0.45, 8.2),
        ("doc_2", "Transformer Architecture Analysis", 31, 25, 0.08, 9.1),
        ("doc_3", "Semantic Similarity Methods", 19, 22, 0.12, 7.5),
        ("doc_4", "Plagiarism Detection Survey", 42, 35, 0.05, 8.8),
        ("doc_5", "Code Clone Detection", 15, 12, 0.18, 6.9),
        ("doc_6", "AI-Generated Text Identification", 28, 30, 0.03, 8.5),
        ("doc_7", "Cross-lingual Transfer Learning", 22, 14, 0.22, 7.2),
        ("doc_8", "Knowledge Graph Completion", 17, 20, 0.10, 7.8),
        ("doc_9", "Information Retrieval Methods", 35, 28, 0.07, 8.9),
        ("doc_10", "Text Summarization Techniques", 20, 16, 0.15, 7.0),
    ]
    journals_map = {
        "doc_1": ["Nature", "IEEE Trans.", "arXiv"],
        "doc_2": ["Science", "ACM Surveys", "NeurIPS"],
        "doc_3": ["JMLR", "ICML", "arXiv"],
        "doc_4": ["Nature", "Science", "ACM Surveys"],
        "doc_5": ["IEEE Trans.", "ICSE", "arXiv"],
        "doc_6": ["Science", "NeurIPS", "ICML"],
        "doc_7": ["ACL", "EMNLP", "arXiv"],
        "doc_8": ["Nature Comm.", "IEEE Trans.", "KDD"],
        "doc_9": ["ACM Surveys", "SIGIR", "Nature"],
        "doc_10": ["JMLR", "ACL", "arXiv"],
    }
    return [
        CitationNetwork(
            doc_id=d[0],
            doc_title=d[1],
            citation_count=d[2],
            cited_by_count=d[3],
            self_citation_rate=d[4],
            avg_source_quality=d[5],
            top_journals=journals_map[d[0]],
            citation_diversity=round(random.uniform(0.4, 0.95), 3),
        )
        for d in docs
    ]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def severity_color(severity: str) -> str:
    return {
        "critical": "#ef4444",
        "high": "#f97316",
        "medium": "#eab308",
        "low": "#22c55e",
    }.get(severity, "#6b7280")


def status_color(status: str) -> str:
    return {
        "open": "#ef4444",
        "investigating": "#f59e0b",
        "resolved": "#22c55e",
        "dismissed": "#94a3b8",
    }.get(status, "#6b7280")


def confidence_color(score: float) -> str:
    if score >= 0.8:
        return "#22c55e"
    if score >= 0.5:
        return "#eab308"
    return "#ef4444"


def render_kpi_card(label: str, value: str, subtitle: str, color: str) -> None:
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


def render_citation_card(citation: Citation) -> None:
    conf_col = confidence_color(citation.confidence_score)
    sim_col = confidence_color(citation.similarity_to_source)
    rel_col = confidence_color(citation.context_relevance)
    flag_badge = (
        '<span style="background:#ef444420;color:#ef4444;padding:2px 8px;border-radius:12px;font-size:10px;font-weight:600;">⚠️ Flagged</span>'
        if citation.flagged
        else ""
    )
    verified_badge = (
        '<span style="background:#22c55e20;color:#22c55e;padding:2px 8px;border-radius:12px;font-size:10px;font-weight:600;">✅ Verified</span>'
        if citation.verified
        else ""
    )

    st.markdown(
        f"""
    <div style="background:rgba(255,255,255,0.06);border-radius:12px;padding:14px;
         border:1px solid rgba(255,255,255,0.08);margin-bottom:10px;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
            <div>
                <span style="font-weight:700;font-size:13px;color:#e2e8f0;">{citation.source_title}</span>
                <span style="font-size:10px;color:#94a3b8;margin-left:8px;">({citation.source_year})</span>
            </div>
            <div>{verified_badge} {flag_badge}</div>
        </div>
        <div style="font-size:11px;color:#94a3b8;margin-bottom:6px;">
            📰 {citation.source_journal} · 👥 {', '.join(citation.source_authors[:2])}
            {'+' + str(len(citation.source_authors) - 2) + ' more' if len(citation.source_authors) > 2 else ''}
        </div>
        <div style="font-size:12px;color:#cbd5e1;margin-bottom:8px;line-height:1.5;font-style:italic;">
            "{citation.citation_context[:200]}..."
        </div>
        <div style="display:flex;gap:12px;font-size:11px;">
            <span>🎯 Confidence: <b style="color:{conf_col}">{citation.confidence_score:.0%}</b></span>
            <span>📝 Similarity: <b style="color:{sim_col}">{citation.similarity_to_source:.0%}</b></span>
            <span>🔗 Relevance: <b style="color:{rel_col}">{citation.context_relevance:.0%}</b></span>
            <span style="color:#94a3b8;">Type: {citation.citation_type.replace('_', ' ').title()}</span>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_anomaly_card(anomaly: CitationAnomaly) -> None:
    sev_col = severity_color(anomaly.severity)
    stat_col = status_color(anomaly.status)
    st.markdown(
        f"""
    <div style="background:rgba(255,255,255,0.06);border-radius:12px;padding:14px;
         border-left:4px solid {sev_col};margin-bottom:10px;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
            <span style="font-weight:700;font-size:13px;color:#e2e8f0;">{anomaly.anomaly_id}: {anomaly.anomaly_type.replace('_', ' ').title()}</span>
            <div style="display:flex;gap:6px;">
                <span style="font-size:10px;padding:2px 8px;border-radius:8px;background:{sev_col}20;color:{sev_col};text-transform:uppercase;font-weight:600;">{anomaly.severity}</span>
                <span style="font-size:10px;padding:2px 8px;border-radius:8px;background:{stat_col}20;color:{stat_col};text-transform:capitalize;font-weight:600;">{anomaly.status}</span>
            </div>
        </div>
        <div style="font-size:12px;color:#cbd5e1;margin-bottom:6px;">{anomaly.description}</div>
        <div style="font-size:11px;color:#94a3b8;">
            📄 Affected: {', '.join(anomaly.affected_docs)} · 🕐 {anomaly.detected_at[:10]}
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_network_node(net: CitationNetwork) -> None:
    bar_width = min(net.citation_count / 50 * 100, 100)
    cited_width = min(net.cited_by_count / 50 * 100, 100)
    self_rate_color = (
        "#ef4444"
        if net.self_citation_rate > 0.3
        else "#eab308"
        if net.self_citation_rate > 0.15
        else "#22c55e"
    )
    st.markdown(
        f"""
    <div style="background:rgba(255,255,255,0.04);border-radius:10px;padding:12px;margin-bottom:8px;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
            <span style="font-weight:700;font-size:13px;color:#e2e8f0;">{net.doc_title}</span>
            <span style="font-size:11px;color:{self_rate_color};font-weight:600;">Self-cite: {net.self_citation_rate:.0%}</span>
        </div>
        <div style="display:flex;gap:16px;font-size:11px;color:#94a3b8;margin-bottom:6px;">
            <span>📤 Citations: {net.citation_count}</span>
            <span>📥 Cited by: {net.cited_by_count}</span>
            <span>⭐ Quality: {net.avg_source_quality}/10</span>
            <span>🎯 Diversity: {net.citation_diversity:.0%}</span>
        </div>
        <div style="display:flex;gap:4px;margin-top:4px;">
            {''.join(f'<span style="font-size:9px;padding:2px 6px;border-radius:8px;background:rgba(139,92,246,0.12);color:#c4b5fd;">{j}</span>' for j in net.top_journals)}
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


# =============================================================================
# MAIN DASHBOARD
# =============================================================================


def render_citation_integrity_dashboard() -> None:
    """Render the full Citation Integrity Dashboard."""
    st.markdown(
        """
    <style>
    .block-container { padding-top: 1rem; }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Generate data
    citations = generate_citations(30)
    authenticity = generate_authenticity(citations)
    anomalies = generate_anomalies()
    network = generate_network()

    # Header
    st.markdown(
        """
    <div style="text-align:center;margin-bottom:20px;">
        <div style="font-size:36px;margin-bottom:8px;">🔍</div>
        <h1 style="font-size:28px;font-weight:800;margin:0;
            background:linear-gradient(135deg,#22c55e,#3b82f6);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            Citation Integrity Dashboard
        </h1>
        <p style="font-size:14px;color:#94a3b8;margin-top:6px;">
            Monitor citation patterns, verify source authenticity, and detect anomalies
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # KPIs
    verified_count = sum(1 for c in citations if c.verified)
    flagged_count = sum(1 for c in citations if c.flagged)
    avg_confidence = sum(c.confidence_score for c in citations) / len(citations)
    open_anomalies = sum(1 for a in anomalies if a.status == "open")
    avg_authenticity = sum(a.authenticity_score for a in authenticity) / len(
        authenticity
    )
    retracted_count = sum(1 for a in authenticity if a.retracted)

    kpi_cols = st.columns(6)
    kpis = [
        ("Citations", str(len(citations)), "total tracked", "#3b82f6"),
        (
            "Verified",
            f"{verified_count}/{len(citations)}",
            f"{verified_count/len(citations):.0%} verified",
            "#22c55e",
        ),
        (
            "Flagged",
            str(flagged_count),
            "need review",
            "#ef4444" if flagged_count > 3 else "#eab308",
        ),
        (
            "Avg Confidence",
            f"{avg_confidence:.0%}",
            "citation confidence",
            confidence_color(avg_confidence),
        ),
        (
            "Anomalies",
            str(open_anomalies),
            "open issues",
            "#ef4444" if open_anomalies > 1 else "#22c55e",
        ),
        (
            "Authenticity",
            f"{avg_authenticity:.0%}",
            f"{retracted_count} retracted",
            confidence_color(avg_authenticity),
        ),
    ]
    for col, (label, value, subtitle, color) in zip(kpi_cols, kpis):
        with col:
            render_kpi_card(label, value, subtitle, color)

    # Tab selection
    tabs = ["📊 Overview", "📋 Citations", "🛡️ Authenticity", "⚠️ Anomalies", "🕸️ Network"]
    selected_tab = st.radio(
        "Dashboard Tabs", tabs, horizontal=True, label_visibility="collapsed"
    )

    if selected_tab == "📊 Overview":
        _render_overview(citations, authenticity, anomalies, network)
    elif selected_tab == "📋 Citations":
        _render_citations_tab(citations)
    elif selected_tab == "🛡️ Authenticity":
        _render_authenticity_tab(authenticity)
    elif selected_tab == "⚠️ Anomalies":
        _render_anomalies_tab(anomalies)
    elif selected_tab == "🕸️ Network":
        _render_network_tab(network, citations)


def _render_overview(
    citations: List[Citation],
    authenticity: List[SourceAuthenticity],
    anomalies: List[CitationAnomaly],
    network: List[CitationNetwork],
) -> None:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📰 Citation Type Distribution")
        type_counts = Counter(c.citation_type for c in citations)
        for ctype, count in type_counts.most_common():
            pct = count / len(citations)
            st.markdown(
                f"""
            <div style="margin-bottom:8px;">
                <div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:3px;">
                    <span style="color:#e2e8f0;font-weight:600;">{ctype.replace('_', ' ').title()}</span>
                    <span style="color:#94a3b8;">{count} ({pct:.0%})</span>
                </div>
                <div style="height:8px;background:rgba(255,255,255,0.08);border-radius:4px;">
                    <div style="height:100%;width:{pct*100}%;background:#3b82f6;border-radius:4px;transition:width 0.8s;"></div>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        st.markdown("#### 📰 Top Cited Journals")
        journal_counts = Counter(c.source_journal for c in citations)
        for journal, count in journal_counts.most_common(5):
            st.markdown(f"• **{journal}**: {count} citations")

    with col2:
        st.markdown("#### 🕐 Source Year Distribution")
        year_counts = Counter(c.source_year for c in citations)
        for year in sorted(year_counts.keys()):
            count = year_counts[year]
            pct = count / max(year_counts.values())
            st.markdown(
                f"""
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
                <span style="font-size:11px;color:#94a3b8;width:36px;">{year}</span>
                <div style="flex:1;height:10px;background:rgba(255,255,255,0.08);border-radius:5px;">
                    <div style="height:100%;width:{pct*100}%;background:#8b5cf6;border-radius:5px;"></div>
                </div>
                <span style="font-size:11px;color:#94a3b8;width:20px;">{count}</span>
            </div>
            """,
                unsafe_allow_html=True,
            )

        st.markdown("#### 🕐 Network Quality Summary")
        avg_quality = sum(n.avg_source_quality for n in network) / len(network)
        avg_diversity = sum(n.citation_diversity for n in network) / len(network)
        avg_self = sum(n.self_citation_rate for n in network) / len(network)
        st.markdown(
            f"""
        <div style="background:rgba(255,255,255,0.04);border-radius:10px;padding:12px;">
            <div style="font-size:12px;color:#cbd5e1;margin-bottom:4px;">⭐ Avg Source Quality: <b>{avg_quality:.1f}/10</b></div>
            <div style="font-size:12px;color:#cbd5e1;margin-bottom:4px;">🎯 Avg Diversity: <b>{avg_diversity:.0%}</b></div>
            <div style="font-size:12px;color:#cbd5e1;">📊 Avg Self-Citation: <b>{avg_self:.0%}</b></div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Anomaly Summary
    st.markdown("#### ⚠️ Recent Anomalies")
    for anomaly in anomalies[:3]:
        render_anomaly_card(anomaly)


def _render_citations_tab(citations: List[Citation]) -> None:
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        type_filter = st.selectbox(
            "Citation Type", ["All"] + list(set(c.citation_type for c in citations))
        )
    with col2:
        status_filter = st.selectbox(
            "Status", ["All", "Verified", "Flagged", "Unverified"]
        )
    with col3:
        sort_by = st.selectbox(
            "Sort By", ["Confidence", "Similarity", "Relevance", "Year"]
        )

    filtered = citations[:]
    if type_filter != "All":
        filtered = [c for c in filtered if c.citation_type == type_filter]
    if status_filter == "Verified":
        filtered = [c for c in filtered if c.verified]
    elif status_filter == "Flagged":
        filtered = [c for c in filtered if c.flagged]
    elif status_filter == "Unverified":
        filtered = [c for c in filtered if not c.verified]

    sort_keys = {
        "Confidence": lambda c: c.confidence_score,
        "Similarity": lambda c: c.similarity_to_source,
        "Relevance": lambda c: c.context_relevance,
        "Year": lambda c: c.source_year,
    }
    filtered.sort(key=sort_keys[sort_by], reverse=True)

    st.markdown(f"**{len(filtered)} citations** match your filters")
    for citation in filtered:
        render_citation_card(citation)


def _render_authenticity_tab(authenticity: List[SourceAuthenticity]) -> None:
    st.markdown("#### 🛡️ Source Authenticity Verification")
    for auth in authenticity:
        risk_badges = ""
        for rf in auth.risk_factors:
            risk_badges += f'<span style="font-size:10px;padding:2px 8px;border-radius:12px;background:#ef444420;color:#ef4444;margin-right:4px;">⚠️ {rf}</span>'

        checks = [
            ("✅ DOI Valid" if auth.doi_valid else "❌ DOI Invalid", auth.doi_valid),
            (
                "✅ Accessible" if auth.url_accessible else "❌ Inaccessible",
                auth.url_accessible,
            ),
            (
                "✅ Metadata Match" if auth.metadata_match else "❌ Mismatch",
                auth.metadata_match,
            ),
            (
                "✅ Peer Reviewed" if auth.peer_reviewed else "⚠️ Not Peer Reviewed",
                auth.peer_reviewed,
            ),
            (
                "✅ Not Retracted" if not auth.retracted else "🚫 RETRACTED",
                not auth.retracted,
            ),
            (
                (
                    "✅ Legitimate Journal"
                    if not auth.predatory_journal
                    else "⚠️ Predatory"
                ),
                not auth.predatory_journal,
            ),
        ]
        checks_html = "".join(
            f'<span style="font-size:10px;color:{"#22c55e" if ok else "#ef4444"};">{label}</span>'
            for label, ok in checks
        )

        st.markdown(
            f"""
        <div style="background:rgba(255,255,255,0.06);border-radius:12px;padding:14px;
             border:1px solid rgba(255,255,255,0.08);margin-bottom:10px;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                <span style="font-weight:700;font-size:13px;color:#e2e8f0;">{auth.title}</span>
                <span style="font-size:14px;font-weight:800;color:{confidence_color(auth.authenticity_score)};">
                    {auth.authenticity_score:.0%}
                </span>
            </div>
            <div style="display:flex;gap:10px;margin-bottom:6px;flex-wrap:wrap;">{checks_html}</div>
            <div style="margin-bottom:4px;">{risk_badges}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )


def _render_anomalies_tab(anomalies: List[CitationAnomaly]) -> None:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("#### ⚠️ Citation Anomalies")
        for anomaly in anomalies:
            render_anomaly_card(anomaly)
    with col2:
        st.markdown("#### 📊 Anomaly Breakdown")
        type_counts = Counter(a.anomaly_type for a in anomalies)
        for atype, count in type_counts.most_common():
            st.markdown(f"• **{atype.replace('_', ' ').title()}**: {count}")

        st.markdown("#### 📊 Severity Distribution")
        sev_counts = Counter(a.severity for a in anomalies)
        for sev, count in sev_counts.most_common():
            col = severity_color(sev)
            st.markdown(
                f"""
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
                <span style="font-size:11px;color:#94a3b8;width:70px;">{sev.title()}</span>
                <div style="flex:1;height:10px;background:rgba(255,255,255,0.08);border-radius:5px;">
                    <div style="height:100%;width:{count/len(anomalies)*100}%;background:{col};border-radius:5px;"></div>
                </div>
                <span style="font-size:11px;color:{col};font-weight:700;">{count}</span>
            </div>
            """,
                unsafe_allow_html=True,
            )

        st.markdown("#### 📊 Status")
        stat_counts = Counter(a.status for a in anomalies)
        for stat, count in stat_counts.most_common():
            col = status_color(stat)
            st.markdown(
                f'<div style="font-size:12px;color:{col};margin-bottom:4px;">{stat.title()}: <b>{count}</b></div>',
                unsafe_allow_html=True,
            )


def _render_network_tab(
    network: List[CitationNetwork], citations: List[Citation]
) -> None:
    st.markdown("#### 🕸️ Citation Network")

    # Network summary
    total_citations = sum(n.citation_count for n in network)
    total_cited_by = sum(n.cited_by_count for n in network)
    avg_self = sum(n.self_citation_rate for n in network) / len(network)

    st.markdown(
        f"""
    <div style="display:flex;gap:16px;margin-bottom:16px;">
        <span style="font-size:12px;color:#94a3b8;">📊 Total Citations: <b style="color:#e2e8f0;">{total_citations}</b></span>
        <span style="font-size:12px;color:#94a3b8;">📥 Total Cited By: <b style="color:#e2e8f0;">{total_cited_by}</b></span>
        <span style="font-size:12px;color:#94a3b8;">📊 Avg Self-Cite: <b style="color:{severity_color('high') if avg_self > 0.2 else '#22c55e'}">{avg_self:.0%}</b></span>
    </div>
    """,
        unsafe_allow_html=True,
    )

    for net in network:
        render_network_node(net)


# =============================================================================
# ENTRY POINT
# =============================================================================


def main():
    """Entry point for the Citation Integrity Dashboard."""
    render_citation_integrity_dashboard()


if __name__ == "__main__":
    main()
