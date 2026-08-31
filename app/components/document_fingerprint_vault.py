"""
Document Fingerprint Vault
===========================
Generates, stores, and compares document fingerprints using multiple hashing
algorithms for plagiarism detection, tamper detection, and similarity search.

Features:
- Multi-algorithm fingerprint generation (MD5, SHA-256, SimHash, MinHash)
- Fingerprint comparison and similarity search
- Tamper detection and integrity verification
- Fingerprint clustering and deduplication
- Historical fingerprint tracking
"""

import hashlib
import math
import random
import string
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
class Fingerprint:
    """A document fingerprint with multiple hash representations."""
    doc_id: str
    doc_title: str
    author: str
    upload_date: str
    file_size: int
    word_count: int
    page_count: int
    md5_hash: str
    sha256_hash: str
    simhash: str
    minhash: List[int]
    shingle_count: int
    ngram_size: int
    algorithm_version: str
    integrity_status: str  # valid, tampered, unknown
    verified_date: Optional[str]


@dataclass
class FingerprintMatch:
    """A match found between two document fingerprints."""
    match_id: str
    source_doc_id: str
    source_title: str
    target_doc_id: str
    target_title: str
    similarity_type: str  # exact, near_duplicate, partial, structural
    similarity_score: float
    matching_algorithm: str
    hamming_distance: int
    jaccard_similarity: float
    shared_shingles: int
    total_shingles: int
    confidence: float
    risk_level: str


@dataclass
class TamperEvent:
    """A detected tampering event."""
    event_id: str
    doc_id: str
    doc_title: str
    tamper_type: str  # content_modification, metadata_tampering, hash_mismatch, time_anomaly
    severity: str
    description: str
    original_hash: str
    current_hash: str
    detected_at: str
    evidence: Dict[str, Any]
    status: str  # open, investigating, resolved


@dataclass
class FingerprintCluster:
    """A cluster of similar document fingerprints."""
    cluster_id: str
    doc_ids: List[str]
    doc_titles: List[str]
    centroid_hash: str
    avg_internal_similarity: float
    cluster_size: int
    risk_level: str
    first_seen: str
    last_updated: str


@dataclass
class HashPerformance:
    """Performance metrics for a hashing algorithm."""
    algorithm: str
    avg_time_ms: float
    collision_rate: float
    fingerprint_size: int
    speed_rank: int
    accuracy_rank: int
    use_case: str


# =============================================================================
# MOCK DATA GENERATORS
# =============================================================================


def generate_hash(length: int = 32) -> str:
    return ''.join(random.choices(string.hexdigits[:16], k=length))


def generate_minhash(size: int = 128) -> List[int]:
    return [random.randint(0, 2**32 - 1) for _ in range(size)]


def generate_fingerprints(count: int = 30) -> List[Fingerprint]:
    titles = [
        "Neural Network Architecture Survey", "Transformer-Based Language Models",
        "Semantic Text Similarity Methods", "Code Clone Detection Algorithms",
        "Plagiarism Detection in Academia", "Cross-Lingual Transfer Learning",
        "Knowledge Graph Construction", "Real-Time Document Processing",
        "Adversarial Text Generation", "Federated Learning for NLP",
        "Graph Neural Networks for Citations", "Multi-Modal Document Understanding",
        "Efficient Fine-Tuning Methods", "Bias Detection in AI Systems",
        "Zero-Shot Classification", "Temporal Text Analysis",
        "Sentiment Analysis Across Domains", "Scientific Paper Summarization",
        "Question Answering Systems", "Spam Detection in Publishing",
        "Plagiarism Obfuscation Analysis", "Semantic Web Technologies",
        "Low-Resource Language Processing", "AI Text Detection Methods",
        "Embedding Model Benchmarks", "Document Layout Analysis",
        "Automated Essay Evaluation", "Named Entity Recognition Survey",
        "Speech-to-Text Comparison", "Multilingual Information Retrieval",
    ]
    authors = ["Smith J.", "Chen W.", "Patel A.", "Kim S.", "Mueller K.", "Garcia M.", "Lee H.", "Brown T."]
    statuses = ["valid", "valid", "valid", "tampered", "valid", "valid"]

    fps = []
    for i in range(count):
        status = random.choice(statuses)
        verified = (datetime.now() - timedelta(days=random.randint(0, 30))).isoformat() if status == "valid" else None
        fps.append(Fingerprint(
            doc_id=f"fp_{i+1:03d}",
            doc_title=titles[i % len(titles)],
            author=random.choice(authors),
            upload_date=(datetime.now() - timedelta(days=random.randint(0, 90))).strftime("%Y-%m-%d"),
            file_size=random.randint(50000, 5000000),
            word_count=random.randint(2000, 25000),
            page_count=random.randint(5, 80),
            md5_hash=generate_hash(32),
            sha256_hash=generate_hash(64),
            simhash=f"{random.randint(0, 2**63):016x}",
            minhash=generate_minhash(64),
            shingle_count=random.randint(50, 500),
            ngram_size=random.choice([3, 4, 5]),
            algorithm_version="2.1",
            integrity_status=status,
            verified_date=verified,
        ))
    return fps


def generate_matches(fps: List[Fingerprint], count: int = 20) -> List[FingerprintMatch]:
    types = ["exact", "near_duplicate", "partial", "structural"]
    algos = ["SimHash", "MinHash", "SHA-256", "Jaccard"]
    matches = []
    for i in range(count):
        s = random.choice(fps)
        t = random.choice([f for f in fps if f.doc_id != s.doc_id])
        mt = random.choice(types)
        sim = random.uniform(0.15, 0.98) if mt != "exact" else random.uniform(0.95, 1.0)
        risk = "critical" if sim > 0.9 else "high" if sim > 0.7 else "medium" if sim > 0.4 else "low"
        shared = random.randint(10, 400)
        total = random.randint(200, 600)
        matches.append(FingerprintMatch(
            match_id=f"FM-{i+1:03d}",
            source_doc_id=s.doc_id, source_title=s.doc_title,
            target_doc_id=t.doc_id, target_title=t.doc_title,
            similarity_type=mt, similarity_score=round(sim, 3),
            matching_algorithm=random.choice(algos),
            hamming_distance=random.randint(0, 50),
            jaccard_similarity=round(random.uniform(0.1, 0.95), 3),
            shared_shingles=shared, total_shingles=total,
            confidence=round(random.uniform(0.5, 0.98), 3),
            risk_level=risk,
        ))
    return sorted(matches, key=lambda m: m.similarity_score, reverse=True)


def generate_tamper_events(fps: List[Fingerprint]) -> List[TamperEvent]:
    tampered = [f for f in fps if f.integrity_status == "tampered"]
    events = []
    for f in tampered[:5]:
        events.append(TamperEvent(
            event_id=f"TE-{uuid.uuid4().hex[:6]}",
            doc_id=f.doc_id, doc_title=f.doc_title,
            tamper_type=random.choice(["content_modification", "metadata_tampering", "hash_mismatch", "time_anomaly"]),
            severity=random.choice(["critical", "high", "medium"]),
            description=f"Document '{f.doc_title}' shows hash mismatch between stored and computed fingerprints.",
            original_hash=f.md5_hash,
            current_hash=generate_hash(32),
            detected_at=(datetime.now() - timedelta(hours=random.randint(1, 48))).isoformat(),
            evidence={"bytes_changed": random.randint(100, 5000), "sections_affected": random.randint(1, 5)},
            status=random.choice(["open", "investigating"]),
        ))
    # Add extra simulated events
    for _ in range(3):
        f = random.choice(fps)
        events.append(TamperEvent(
            event_id=f"TE-{uuid.uuid4().hex[:6]}",
            doc_id=f.doc_id, doc_title=f.doc_title,
            tamper_type=random.choice(["content_modification", "time_anomaly"]),
            severity=random.choice(["medium", "low"]),
            description=f"Suspected content modification in '{f.doc_title}'. Timestamps inconsistent.",
            original_hash=f.md5_hash,
            current_hash=generate_hash(32),
            detected_at=(datetime.now() - timedelta(hours=random.randint(1, 72))).isoformat(),
            evidence={"bytes_changed": random.randint(50, 2000)},
            status=random.choice(["open", "resolved"]),
        ))
    return events


def generate_clusters(fps: List[Fingerprint]) -> List[FingerprintCluster]:
    clusters = []
    for i in range(5):
        n = random.randint(2, 5)
        selected = random.sample(fps, min(n, len(fps)))
        clusters.append(FingerprintCluster(
            cluster_id=f"CL-{i+1:03d}",
            doc_ids=[f.doc_id for f in selected],
            doc_titles=[f.doc_title for f in selected],
            centroid_hash=generate_hash(32),
            avg_internal_similarity=round(random.uniform(0.5, 0.95), 3),
            cluster_size=len(selected),
            risk_level="high" if random.random() > 0.5 else "medium",
            first_seen=(datetime.now() - timedelta(days=random.randint(10, 60))).isoformat(),
            last_updated=(datetime.now() - timedelta(days=random.randint(0, 10))).isoformat(),
        ))
    return clusters


def generate_hash_performance() -> List[HashPerformance]:
    return [
        HashPerformance("MD5", 0.8, 0.001, 128, 1, 3, "Fast checksum verification"),
        HashPerformance("SHA-256", 2.5, 0.0001, 256, 3, 1, "Cryptographic integrity"),
        HashPerformance("SimHash", 5.2, 0.05, 64, 2, 2, "Near-duplicate detection"),
        HashPerformance("MinHash", 8.1, 0.03, 512, 4, 2, "Large-scale similarity"),
        HashPerformance("Locality-Sensitive Hash", 6.5, 0.04, 128, 5, 2, "Approximate nearest neighbor"),
    ]


# =============================================================================
# HELPERS
# =============================================================================


def status_color(s: str) -> str:
    return {"valid": "#22c55e", "tampered": "#ef4444", "unknown": "#94a3b8"}.get(s, "#6b7280")


def sim_color(s: float) -> str:
    if s > 0.9: return "#ef4444"
    if s > 0.7: return "#f97316"
    if s > 0.4: return "#eab308"
    return "#22c55e"


def severity_color(s: str) -> str:
    return {"critical": "#ef4444", "high": "#f97316", "medium": "#eab308", "low": "#22c55e"}.get(s, "#6b7280")


def format_size(b: int) -> str:
    if b >= 1000000: return f"{b/1000000:.1f} MB"
    if b >= 1000: return f"{b/1000:.1f} KB"
    return f"{b} B"


def render_kpi(label: str, value: str, subtitle: str, color: str) -> None:
    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.06);border-radius:14px;padding:18px 14px;
         border:1px solid rgba(255,255,255,0.08);text-align:center;">
        <div style="font-size:26px;font-weight:800;color:{color};margin-bottom:4px;">{value}</div>
        <div style="font-size:12px;font-weight:600;color:#e2e8f0;margin-bottom:2px;">{label}</div>
        <div style="font-size:10px;color:#94a3b8;">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)


def render_fingerprint_card(fp: Fingerprint, expanded: bool = False) -> None:
    ic = status_color(fp.integrity_status)
    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.06);border-radius:12px;padding:14px;
         border-left:4px solid {ic};margin-bottom:10px;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
            <div>
                <span style="font-weight:700;font-size:13px;color:#e2e8f0;">{fp.doc_title}</span>
                <span style="font-size:11px;color:#94a3b8;margin-left:8px;">by {fp.author}</span>
            </div>
            <span style="font-size:10px;padding:2px 10px;border-radius:12px;background:{ic}20;color:{ic};text-transform:uppercase;font-weight:600;">{fp.integrity_status}</span>
        </div>
        <div style="display:flex;gap:12px;font-size:11px;color:#94a3b8;margin-bottom:6px;">
            <span>📄 {fp.word_count:,} words</span>
            <span>📑 {fp.page_count} pages</span>
            <span>💾 {format_size(fp.file_size)}</span>
            <span>📅 {fp.upload_date}</span>
            <span>🔢 {fp.shingle_count} shingles</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if expanded:
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.04);border-radius:8px;padding:10px;margin-top:6px;">
            <div style="font-size:11px;color:#94a3b8;margin-bottom:4px;"><b>MD5:</b> <span style="font-family:monospace;color:#e2e8f0;">{fp.md5_hash[:32]}...</span></div>
            <div style="font-size:11px;color:#94a3b8;margin-bottom:4px;"><b>SHA-256:</b> <span style="font-family:monospace;color:#e2e8f0;">{fp.sha256_hash[:48]}...</span></div>
            <div style="font-size:11px;color:#94a3b8;margin-bottom:4px;"><b>SimHash:</b> <span style="font-family:monospace;color:#e2e8f0;">{fp.simhash}</span></div>
            <div style="font-size:11px;color:#94a3b8;margin-bottom:4px;"><b>MinHash:</b> <span style="font-family:monospace;color:#e2e8f0;">[{', '.join(str(h)[:8] for h in fp.minhash[:4])}...]</span></div>
            <div style="font-size:11px;color:#94a3b8;"><b>N-gram:</b> {fp.ngram_size}-shingles · <b>Algorithm:</b> v{fp.algorithm_version} · <b>Verified:</b> {fp.verified_date[:10] if fp.verified_date else 'Never'}</div>
        </div>
        """, unsafe_allow_html=True)


def render_match_card(m: FingerprintMatch) -> None:
    sc = sim_color(m.similarity_score)
    type_colors = {"exact": "#ef4444", "near_duplicate": "#f97316", "partial": "#eab308", "structural": "#8b5cf6"}
    tc = type_colors.get(m.similarity_type, "#6b7280")
    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.06);border-radius:12px;padding:14px;
         border-left:4px solid {sc};margin-bottom:10px;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
            <span style="font-weight:700;font-size:13px;color:#e2e8f0;">{m.match_id}</span>
            <div style="display:flex;gap:6px;">
                <span style="font-size:10px;padding:2px 8px;border-radius:8px;background:{tc}20;color:{tc};font-weight:600;">{m.similarity_type.replace('_', ' ').title()}</span>
                <span style="font-size:10px;padding:2px 8px;border-radius:8px;background:{sc}20;color:{sc};text-transform:uppercase;font-weight:600;">{m.risk_level}</span>
            </div>
        </div>
        <div style="font-size:12px;color:#cbd5e1;margin-bottom:4px;">
            📄 {m.source_title[:40]}... → 📄 {m.target_title[:40]}...
        </div>
        <div style="display:flex;gap:12px;font-size:11px;color:#94a3b8;margin-bottom:6px;">
            <span>📊 {m.similarity_score:.0%} similarity</span>
            <span>🔢 {m.hamming_distance} hamming</span>
            <span>🎯 {m.jaccard_similarity:.0%} Jaccard</span>
            <span>🧩 {m.shared_shingles}/{m.total_shingles} shingles</span>
            <span>⚙️ {m.matching_algorithm}</span>
        </div>
        <div style="height:8px;background:rgba(255,255,255,0.08);border-radius:4px;">
            <div style="height:100%;width:{m.similarity_score*100}%;background:{sc};border-radius:4px;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_tamper_card(te: TamperEvent) -> None:
    sc = severity_color(te.severity)
    status_colors = {"open": "#ef4444", "investigating": "#f59e0b", "resolved": "#22c55e"}
    stc = status_colors.get(te.status, "#6b7280")
    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.06);border-radius:12px;padding:14px;
         border-left:4px solid {sc};margin-bottom:10px;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
            <span style="font-weight:700;font-size:13px;color:#e2e8f0;">{te.event_id}: {te.tamper_type.replace('_', ' ').title()}</span>
            <div style="display:flex;gap:6px;">
                <span style="font-size:10px;padding:2px 8px;border-radius:8px;background:{sc}20;color:{sc};text-transform:uppercase;font-weight:600;">{te.severity}</span>
                <span style="font-size:10px;padding:2px 8px;border-radius:8px;background:{stc}20;color:{stc};text-transform:capitalize;font-weight:600;">{te.status}</span>
            </div>
        </div>
        <div style="font-size:12px;color:#cbd5e1;margin-bottom:4px;">{te.description}</div>
        <div style="font-size:11px;color:#94a3b8;margin-bottom:4px;">📄 {te.doc_title} · 🕐 {te.detected_at[:16]}</div>
        <div style="font-size:10px;color:#94a3b8;font-family:monospace;background:rgba(255,255,255,0.04);padding:6px;border-radius:6px;">
            Original: {te.original_hash[:20]}... → Current: {te.current_hash[:20]}...
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# MAIN DASHBOARD
# =============================================================================


def render_fingerprint_vault() -> None:
    st.markdown("""
    <style>
    .block-container { padding-top: 1rem; }
    </style>
    """, unsafe_allow_html=True)

    fps = generate_fingerprints(30)
    matches = generate_matches(fps, 20)
    tampers = generate_tamper_events(fps)
    clusters = generate_clusters(fps)
    perf = generate_hash_performance()

    # Header
    st.markdown("""
    <div style="text-align:center;margin-bottom:20px;">
        <div style="font-size:36px;margin-bottom:8px;">🔐</div>
        <h1 style="font-size:28px;font-weight:800;margin:0;
            background:linear-gradient(135deg,#06b6d4,#8b5cf6);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            Document Fingerprint Vault
        </h1>
        <p style="font-size:14px;color:#94a3b8;margin-top:6px;">
            Generate, compare, and verify document fingerprints for integrity
        </p>
    </div>
    """, unsafe_allow_html=True)

    # KPIs
    total_fps = len(fps)
    valid_fps = sum(1 for f in fps if f.integrity_status == "valid")
    tampered_fps = sum(1 for f in fps if f.integrity_status == "tampered")
    total_matches = len(matches)
    exact_matches = sum(1 for m in matches if m.similarity_type == "exact")
    open_tampers = sum(1 for t in tampers if t.status == "open")

    cols = st.columns(6)
    kpis = [
        ("Fingerprints", str(total_fps), "in vault", "#06b6d4"),
        ("Valid", str(valid_fps), f"{valid_fps/total_fps:.0%} integrity", "#22c55e"),
        ("Tampered", str(tampered_fps), "need investigation", "#ef4444"),
        ("Matches Found", str(total_matches), "similarities detected", "#8b5cf6"),
        ("Exact Matches", str(exact_matches), "potential duplicates", "#ef4444" if exact_matches > 2 else "#eab308"),
        ("Open Tampers", str(open_tampers), "events to review", "#ef4444" if open_tampers > 1 else "#22c55e"),
    ]
    for col, (l, v, s, c) in zip(cols, kpis):
        with col:
            render_kpi(l, v, s, c)

    # Tabs
    tabs = ["🔐 Vault", "🔍 Matches", "🛡️ Tamper", "📊 Clusters", "⚡ Performance"]
    selected = st.radio("Tabs", tabs, horizontal=True, label_visibility="collapsed")

    if selected == "🔐 Vault":
        _render_vault(fps)
    elif selected == "🔍 Matches":
        _render_matches(matches)
    elif selected == "🛡️ Tamper":
        _render_tamper(tampers)
    elif selected == "📊 Clusters":
        _render_clusters(clusters, fps)
    elif selected == "⚡ Performance":
        _render_performance(perf)


def _render_vault(fps):
    col1, col2, col3 = st.columns(3)
    with col1:
        status_filter = st.selectbox("Integrity Status", ["All", "Valid", "Tampered", "Unknown"])
    with col2:
        sort_by = st.selectbox("Sort By", ["Upload Date", "Word Count", "File Size", "Shingle Count"])
    with col3:
        search = st.text_input("🔍 Search documents...")

    filtered = fps[:]
    if status_filter != "All":
        filtered = [f for f in filtered if f.integrity_status == status_filter.lower()]
    if search:
        filtered = [f for f in filtered if search.lower() in f.doc_title.lower() or search.lower() in f.author.lower()]

    sort_map = {"Upload Date": lambda f: f.upload_date, "Word Count": lambda f: f.word_count, "File Size": lambda f: f.file_size, "Shingle Count": lambda f: f.shingle_count}
    filtered.sort(key=sort_map[sort_by], reverse=True)

    st.markdown(f"**{len(filtered)} fingerprints** in vault")

    for fp in filtered:
        with st.expander(f"{fp.doc_title} — {fp.integrity_status.title()}", expanded=False):
            render_fingerprint_card(fp, expanded=True)


def _render_matches(matches):
    col1, col2, col3 = st.columns(3)
    with col1:
        type_filter = st.selectbox("Match Type", ["All", "Exact", "Near Duplicate", "Partial", "Structural"])
    with col2:
        risk_filter = st.selectbox("Risk Level", ["All", "Critical", "High", "Medium", "Low"])
    with col3:
        sort_by = st.selectbox("Sort By", ["Similarity", "Jaccard", "Hamming Distance"])

    filtered = matches[:]
    if type_filter != "All":
        filtered = [m for m in filtered if m.similarity_type == type_filter.lower().replace(' ', '_')]
    if risk_filter != "All":
        filtered = [m for m in filtered if m.risk_level == risk_filter.lower()]

    sort_map = {"Similarity": lambda m: m.similarity_score, "Jaccard": lambda m: m.jaccard_similarity, "Hamming Distance": lambda m: m.hamming_distance}
    filtered.sort(key=sort_map[sort_by], reverse=(sort_by != "Hamming Distance"))

    st.markdown(f"**{len(filtered)} matches** found")
    for m in filtered:
        render_match_card(m)


def _render_tamper(tampers):
    st.markdown("#### 🛡️ Tamper Detection Events")
    col1, col2 = st.columns([2, 1])
    with col1:
        for te in tampers:
            render_tamper_card(te)
    with col2:
        st.markdown("#### 📊 Event Statistics")
        sev_counts = Counter(t.severity for t in tampers)
        for sev, count in sev_counts.most_common():
            sc = severity_color(sev)
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
                <span style="font-size:11px;color:#94a3b8;width:70px;">{sev.title()}</span>
                <div style="flex:1;height:10px;background:rgba(255,255,255,0.08);border-radius:5px;">
                    <div style="height:100%;width:{count/len(tampers)*100}%;background:{sc};border-radius:5px;"></div>
                </div>
                <span style="font-size:11px;color:{sc};font-weight:700;">{count}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("#### 📊 Type Distribution")
        type_counts = Counter(t.tamper_type for t in tampers)
        for tt, count in type_counts.most_common():
            st.markdown(f"• **{tt.replace('_', ' ').title()}**: {count}")

        st.markdown("#### 📊 Status")
        stat_counts = Counter(t.status for t in tampers)
        for st_, count in stat_counts.most_common():
            stc = {"open": "#ef4444", "investigating": "#f59e0b", "resolved": "#22c55e"}.get(st_, "#6b7280")
            st.markdown(f'<div style="font-size:12px;color:{stc};margin-bottom:4px;">{st_.title()}: <b>{count}</b></div>', unsafe_allow_html=True)


def _render_clusters(clusters, fps):
    st.markdown("#### 📊 Fingerprint Clusters")
    for cl in clusters:
        rc = sim_color(cl.avg_internal_similarity)
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.06);border-radius:12px;padding:14px;
             border:1px solid rgba(255,255,255,0.08);margin-bottom:10px;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                <span style="font-weight:700;font-size:13px;color:#e2e8f0;">{cl.cluster_id}</span>
                <div style="display:flex;gap:6px;">
                    <span style="font-size:10px;padding:2px 8px;border-radius:8px;background:{severity_color(cl.risk_level)}20;color:{severity_color(cl.risk_level)};text-transform:uppercase;font-weight:600;">{cl.risk_level}</span>
                    <span style="font-size:10px;padding:2px 8px;border-radius:8px;background:rgba(255,255,255,0.06);color:#94a3b8;">{cl.cluster_size} docs</span>
                </div>
            </div>
            <div style="font-size:12px;color:#94a3b8;margin-bottom:4px;">
                📊 Avg Similarity: <b style="color:{rc}">{cl.avg_internal_similarity:.0%}</b> ·
                🕐 {cl.first_seen[:10]} → {cl.last_updated[:10]}
            </div>
            <div style="display:flex;flex-wrap:wrap;gap:4px;">
                {''.join(f'<span style="font-size:10px;padding:2px 8px;border-radius:12px;background:rgba(139,92,246,0.12);color:#c4b5fd;">{t[:35]}...</span>' for t in cl.doc_titles)}
            </div>
        </div>
        """, unsafe_allow_html=True)


def _render_performance(perf):
    st.markdown("#### ⚡ Hash Algorithm Performance")
    for p in perf:
        speed_pct = (6 - p.speed_rank) / 5 * 100
        acc_pct = (6 - p.accuracy_rank) / 5 * 100
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.06);border-radius:12px;padding:14px;
             border:1px solid rgba(255,255,255,0.08);margin-bottom:10px;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
                <span style="font-weight:700;font-size:14px;color:#e2e8f0;">{p.algorithm}</span>
                <span style="font-size:11px;color:#94a3b8;">{p.fingerprint_size}-bit · {p.use_case}</span>
            </div>
            <div style="display:flex;gap:16px;margin-bottom:6px;">
                <span style="font-size:12px;color:#94a3b8;">⏱ {p.avg_time_ms}ms avg</span>
                <span style="font-size:12px;color:#94a3b8;">💥 {p.collision_rate:.1%} collision</span>
                <span style="font-size:12px;color:#94a3b8;">📊 Speed rank: #{p.speed_rank}</span>
                <span style="font-size:12px;color:#94a3b8;">🎯 Accuracy rank: #{p.accuracy_rank}</span>
            </div>
            <div style="display:flex;gap:12px;">
                <div style="flex:1;">
                    <div style="font-size:10px;color:#94a3b8;margin-bottom:2px;">Speed</div>
                    <div style="height:8px;background:rgba(255,255,255,0.08);border-radius:4px;">
                        <div style="height:100%;width:{speed_pct}%;background:#3b82f6;border-radius:4px;"></div>
                    </div>
                </div>
                <div style="flex:1;">
                    <div style="font-size:10px;color:#94a3b8;margin-bottom:2px;">Accuracy</div>
                    <div style="height:8px;background:rgba(255,255,255,0.08);border-radius:4px;">
                        <div style="height:100%;width:{acc_pct}%;background:#22c55e;border-radius:4px;"></div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# ENTRY POINT
# =============================================================================


def main():
    render_fingerprint_vault()


if __name__ == "__main__":
    main()
