"""
AI Content Authenticator
========================
Detects AI-generated content, deepfake text, and synthetic media with
confidence scoring, provenance tracking, and model fingerprinting.

Features:
- AI-generated content detection with confidence scoring
- Deepfake text identification
- Content provenance and origin tracking
- AI model fingerprinting
- Statistical anomaly detection for synthetic text
"""

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
class ContentAuthenticity:
    """Authenticity assessment for a piece of content."""
    doc_id: str
    doc_title: str
    author: str
    content_type: str  # essay, article, code, report, email, abstract
    word_count: int
    language: str
    authenticity_score: float  # 0-100 (higher = more human)
    ai_probability: float  # 0-1
    detected_models: List[str]
    detection_methods: List[str]
    statistical_features: StatisticalFeatures
    provenance: ContentProvenance
    risk_factors: List[str]
    recommendations: List[str]
    scan_date: str
    scan_duration: float


@dataclass
class StatisticalFeatures:
    """Statistical features used for AI detection."""
    perplexity: float
    burstiness: float  # measure of sentence length variation
    vocabulary_richness: float
    repetition_index: float
    syntactic_complexity: float
    entropy: float
    ngram_diversity: float
    sentence_length_cv: float  # coefficient of variation
    punctuation_density: float
    stopword_ratio: float


@dataclass
class ContentProvenance:
    """Provenance and origin tracking for content."""
    origin_method: str  # ai_generated, ai_assisted, human_written, hybrid, unknown
    confidence: float
    creation_timestamp: Optional[str]
    editing_history: List[Dict[str, Any]]
    metadata_signature: str
    platform_detected: Optional[str]
    watermark_detected: bool
    ai_watermark_present: bool
    provenance_chain: List[str]


@dataclass
class DeepfakeDetection:
    """Detection result for deepfake/synthetic text."""
    detection_id: str
    doc_id: str
    doc_title: str
    deepfake_type: str  # full_ai, parrot, style_transfer, template_fill, hybrid
    confidence: float
    indicators: List[str]
    model_fingerprint: Optional[str]
    generation_method: str
    stealth_level: str  # low, medium, high, expert
    recommended_action: str
    detected_at: str


@dataclass
class AIModelFingerprint:
    """Fingerprint of an AI model's output patterns."""
    model_name: str
    model_family: str  # GPT, Claude, LLaMA, PaLM, Gemini, etc.
    signature_patterns: List[str]
    avg_perplexity: float
    avg_burstiness: float
    typical_vocabulary: List[str]
    output_characteristics: Dict[str, Any]
    detection_accuracy: float
    samples_analyzed: int


@dataclass
class DetectionTimeline:
    """Timeline entry for detection events."""
    date: str
    total_scanned: int
    ai_detected: int
    human_verified: int
    deepfake_detected: int
    avg_confidence: float


# =============================================================================
# MOCK DATA GENERATORS
# =============================================================================


def generate_content_authenticity(count: int = 25) -> List[ContentAuthenticity]:
    titles = [
        "The Future of Artificial Intelligence in Healthcare",
        "Climate Change: A Comprehensive Analysis",
        "Machine Learning for Drug Discovery",
        "Quantum Computing Breakthroughs in 2026",
        "The Ethics of Autonomous Vehicles",
        "Neural Architecture Search: A Survey",
        "Sustainable Energy Solutions for Urban Areas",
        "Blockchain in Supply Chain Management",
        "Natural Language Processing Advances",
        "Cybersecurity Threats in the AI Era",
        "The Impact of Remote Work on Productivity",
        "Deep Learning for Medical Imaging",
        "Renewable Energy Grid Optimization",
        "Federated Learning Privacy Guarantees",
        "The Future of Digital Education",
        "Space Exploration with AI Robotics",
        "Urban Planning with Smart City Data",
        "The Psychology of Decision Making",
        "Genomics and Personalized Medicine",
        "Ocean Conservation Technology",
        "AI-Generated Art: Creativity vs Originality",
        "The Rise of Autonomous Systems",
        "Data Privacy in the Modern World",
        "The Evolution of Search Algorithms",
        "Biotechnology and Agricultural Innovation",
    ]
    authors = ["Smith J.", "Chen W.", "Patel A.", "Kim S.", "Mueller K.", "Garcia M.", "AI-Generated", "ChatGPT Output", "Unknown Author"]
    content_types = ["essay", "article", "report", "abstract", "code", "email"]
    models = ["GPT-4", "Claude 3", "Gemini Pro", "LLaMA 3", "PaLM 2", "Mistral", "DeepSeek"]

    docs = []
    for i in range(count):
        auth_score = random.uniform(5, 98)
        ai_prob = 1 - auth_score / 100
        detected = random.sample(models, random.randint(0, 3)) if ai_prob > 0.5 else []

        methods = []
        if ai_prob > 0.3: methods.append("Perplexity Analysis")
        if ai_prob > 0.5: methods.append("Burstiness Detection")
        if ai_prob > 0.4: methods.append("N-gram Analysis")
        if ai_prob > 0.6: methods.append("Watermark Detection")
        if ai_prob > 0.7: methods.append("Stylometric Analysis")

        features = StatisticalFeatures(
            perplexity=round(random.uniform(10, 200) if ai_prob > 0.5 else random.uniform(50, 500), 1),
            burstiness=round(random.uniform(0.1, 0.4) if ai_prob > 0.5 else random.uniform(0.3, 0.8), 3),
            vocabulary_richness=round(random.uniform(0.3, 0.6) if ai_prob > 0.5 else random.uniform(0.4, 0.85), 3),
            repetition_index=round(random.uniform(0.15, 0.45) if ai_prob > 0.5 else random.uniform(0.05, 0.2), 3),
            syntactic_complexity=round(random.uniform(0.2, 0.5) if ai_prob > 0.5 else random.uniform(0.3, 0.8), 3),
            entropy=round(random.uniform(3.5, 5.5) if ai_prob > 0.5 else random.uniform(5.0, 8.0), 3),
            ngram_diversity=round(random.uniform(0.3, 0.6) if ai_prob > 0.5 else random.uniform(0.5, 0.9), 3),
            sentence_length_cv=round(random.uniform(0.15, 0.35) if ai_prob > 0.5 else random.uniform(0.3, 0.7), 3),
            punctuation_density=round(random.uniform(0.05, 0.12) if ai_prob > 0.5 else random.uniform(0.06, 0.15), 3),
            stopword_ratio=round(random.uniform(0.4, 0.6) if ai_prob > 0.5 else random.uniform(0.3, 0.55), 3),
        )

        prov_method = "ai_generated" if ai_prob > 0.8 else "ai_assisted" if ai_prob > 0.5 else "human_written" if ai_prob < 0.2 else "hybrid"
        provenance = ContentProvenance(
            origin_method=prov_method, confidence=round(random.uniform(0.6, 0.98), 3),
            creation_timestamp=(datetime.now() - timedelta(days=random.randint(0, 30))).isoformat(),
            editing_history=[{"action": "created", "timestamp": datetime.now().isoformat()}],
            metadata_signature=uuid.uuid4().hex[:16],
            platform_detected=random.choice(["Web", "Desktop App", "API", None]),
            watermark_detected=random.random() > 0.7,
            ai_watermark_present=random.random() > 0.6 if ai_prob > 0.5 else False,
            provenance_chain=["original"] if ai_prob < 0.3 else ["original", "ai_processing", "review"],
        )

        risk_factors = []
        if ai_prob > 0.7: risk_factors.append("High AI probability detected")
        if features.burstiness < 0.3: risk_factors.append("Low burstiness suggests AI generation")
        if features.repetition_index > 0.3: risk_factors.append("High repetition index")
        if not provenance.watermark_detected and ai_prob > 0.5: risk_factors.append("No content watermark")

        recommendations = []
        if ai_prob > 0.8: recommendations.append("Flag for manual review — likely AI-generated")
        elif ai_prob > 0.5: recommendations.append("Investigate further — possible AI-assisted content")
        elif ai_prob < 0.2: recommendations.append("Content appears authentic")

        docs.append(ContentAuthenticity(
            doc_id=f"doc_{i+1:03d}", doc_title=titles[i % len(titles)],
            author=random.choice(authors), content_type=random.choice(content_types),
            word_count=random.randint(500, 8000), language="English",
            authenticity_score=round(auth_score, 1), ai_probability=round(ai_prob, 3),
            detected_models=detected, detection_methods=methods,
            statistical_features=features, provenance=provenance,
            risk_factors=risk_factors, recommendations=recommendations,
            scan_date=(datetime.now() - timedelta(hours=random.randint(1, 48))).isoformat(),
            scan_duration=round(random.uniform(1, 15), 1),
        ))
    return sorted(docs, key=lambda d: d.ai_probability, reverse=True)


def generate_deepfake_detections(docs: List[ContentAuthenticity]) -> List[DeepfakeDetection]:
    detections = []
    ai_docs = [d for d in docs if d.ai_probability > 0.5][:8]
    for d in ai_docs:
        stealth = random.choice(["low", "medium", "high", "expert"])
        detections.append(DeepfakeDetection(
            detection_id=f"DF-{uuid.uuid4().hex[:6]}",
            doc_id=d.doc_id, doc_title=d.doc_title,
            deepfake_type=random.choice(["full_ai", "parrot", "style_transfer", "template_fill", "hybrid"]),
            confidence=round(random.uniform(0.5, 0.98), 3),
            indicators=random.sample([
                "Unusual token distribution", "Consistent sentence length",
                "Low perplexity scores", "Missing personal voice",
                "Generic transitions", "Lack of specific examples",
                "Uniform paragraph structure", "Over-formal register",
            ], random.randint(2, 5)),
            model_fingerprint=random.choice(d.detected_models + [None]),
            generation_method=random.choice(["API generation", "Web interface", "Fine-tuned model", "Prompt engineering"]),
            stealth_level=stealth,
            recommended_action="Immediate review" if stealth in ("high", "expert") else "Standard review",
            detected_at=(datetime.now() - timedelta(hours=random.randint(1, 72))).isoformat(),
        ))
    return detections


def generate_model_fingerprints() -> List[AIModelFingerprint]:
    return [
        AIModelFingerprint("GPT-4", "GPT", ["consistent tone", "detailed explanations", "structured output"],
                          35.2, 0.22, ["furthermore", "additionally", "consequently"], {"avg_tokens_per_sentence": 22, "formality": 0.85}, 0.92, 500),
        AIModelFingerprint("Claude 3", "Claude", ["nuanced reasoning", "balanced perspective", "cautious hedging"],
                          42.1, 0.28, ["however", "it's worth noting", "in contrast"], {"avg_tokens_per_sentence": 25, "formality": 0.80}, 0.88, 350),
        AIModelFingerprint("Gemini Pro", "Gemini", ["concise responses", "fact-focused", "direct answers"],
                          38.5, 0.25, ["specifically", "in particular", "notably"], {"avg_tokens_per_sentence": 20, "formality": 0.82}, 0.85, 280),
        AIModelFingerprint("LLaMA 3", "LLaMA", ["informal tone", "code-heavy", "technical vocabulary"],
                          45.3, 0.30, ["basically", "essentially", "typically"], {"avg_tokens_per_sentence": 18, "formality": 0.65}, 0.78, 200),
        AIModelFingerprint("Mistral", "Mistral", ["European English patterns", "formal register", "detailed analysis"],
                          40.8, 0.26, ["therefore", "hence", "thus"], {"avg_tokens_per_sentence": 23, "formality": 0.88}, 0.82, 150),
    ]


def generate_timeline(days: int = 30) -> List[DetectionTimeline]:
    return [
        DetectionTimeline(
            date=(datetime.now() - timedelta(days=days - i)).strftime("%Y-%m-%d"),
            total_scanned=random.randint(10, 50),
            ai_detected=random.randint(2, 15),
            human_verified=random.randint(5, 30),
            deepfake_detected=random.randint(0, 5),
            avg_confidence=round(random.uniform(0.6, 0.95), 3),
        ) for i in range(days)
    ]


# =============================================================================
# HELPERS
# =============================================================================


def auth_color(score: float) -> str:
    if score > 75: return "#22c55e"
    if score > 50: return "#eab308"
    if score > 25: return "#f97316"
    return "#ef4444"


def stealth_color(s: str) -> str:
    return {"low": "#22c55e", "medium": "#eab308", "high": "#f97316", "expert": "#ef4444"}.get(s, "#6b7280")


def render_kpi(label: str, value: str, subtitle: str, color: str) -> None:
    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.06);border-radius:14px;padding:18px 14px;
         border:1px solid rgba(255,255,255,0.08);text-align:center;">
        <div style="font-size:26px;font-weight:800;color:{color};margin-bottom:4px;">{value}</div>
        <div style="font-size:12px;font-weight:600;color:#e2e8f0;margin-bottom:2px;">{label}</div>
        <div style="font-size:10px;color:#94a3b8;">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)


def render_auth_card(doc: ContentAuthenticity, expanded: bool = False) -> None:
    ac = auth_color(doc.authenticity_score)
    ai_c = "#ef4444" if doc.ai_probability > 0.7 else "#f97316" if doc.ai_probability > 0.4 else "#22c55e"
    prov_colors = {"ai_generated": "#ef4444", "ai_assisted": "#f97316", "hybrid": "#eab308", "human_written": "#22c55e", "unknown": "#94a3b8"}
    pc = prov_colors.get(doc.provenance.origin_method, "#94a3b8")

    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.06);border-radius:12px;padding:14px;
         border-left:4px solid {ac};margin-bottom:10px;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
            <div>
                <span style="font-weight:700;font-size:13px;color:#e2e8f0;">{doc.doc_title}</span>
                <span style="font-size:11px;color:#94a3b8;margin-left:8px;">by {doc.author}</span>
            </div>
            <div style="text-align:right;">
                <div style="font-size:18px;font-weight:800;color:{ac};">{doc.authenticity_score:.0f}%</div>
                <div style="font-size:9px;color:#94a3b8;">authentic</div>
            </div>
        </div>
        <div style="display:flex;gap:10px;font-size:11px;color:#94a3b8;margin-bottom:6px;">
            <span>🤖 AI: <b style="color:{ai_c}">{doc.ai_probability:.0%}</b></span>
            <span>📄 {doc.word_count:,} words</span>
            <span>📝 {doc.content_type.title()}</span>
            <span>🏷️ <span style="color:{pc};text-transform:capitalize;font-weight:600;">{doc.provenance.origin_method.replace('_', ' ')}</span></span>
            {f'<span>🔒 Watermark: {"✅" if doc.provenance.watermark_detected else "❌"}</span>' if doc.provenance.ai_watermark_present else ''}
        </div>
        {f'<div style="display:flex;gap:4px;flex-wrap:wrap;margin-bottom:4px;">' + "".join(f'<span style="font-size:9px;padding:2px 6px;border-radius:8px;background:rgba(139,92,246,0.12);color:#c4b5fd;">{m}</span>' for m in doc.detected_models) + '</div>' if doc.detected_models else ''}
        {"".join(f'<div style="font-size:11px;color:#f97316;margin-bottom:2px;">⚠️ {rf}</div>' for rf in doc.risk_factors[:2]) if doc.risk_factors else ''}
    </div>
    """, unsafe_allow_html=True)

    if expanded:
        sf = doc.statistical_features
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.04);border-radius:8px;padding:10px;margin-top:6px;">
            <div style="font-size:12px;font-weight:700;color:#94a3b8;margin-bottom:6px;">📊 Statistical Features:</div>
            <div style="display:grid;grid-template-columns:repeat(5, 1fr);gap:6px;">
                <div style="text-align:center;"><div style="font-size:14px;font-weight:800;color:#e2e8f0;">{sf.perplexity:.0f}</div><div style="font-size:9px;color:#94a3b8;">Perplexity</div></div>
                <div style="text-align:center;"><div style="font-size:14px;font-weight:800;color:{auth_color(sf.burstiness*100)};">{sf.burstiness:.2f}</div><div style="font-size:9px;color:#94a3b8;">Burstiness</div></div>
                <div style="text-align:center;"><div style="font-size:14px;font-weight:800;color:#e2e8f0;">{sf.vocabulary_richness:.2f}</div><div style="font-size:9px;color:#94a3b8;">Vocab Richness</div></div>
                <div style="text-align:center;"><div style="font-size:14px;font-weight:800;color:#e2e8f0;">{sf.repetition_index:.2f}</div><div style="font-size:9px;color:#94a3b8;">Repetition</div></div>
                <div style="text-align:center;"><div style="font-size:14px;font-weight:800;color:#e2e8f0;">{sf.entropy:.1f}</div><div style="font-size:9px;color:#94a3b8;">Entropy</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if doc.recommendations:
            st.markdown("**Recommendations:**")
            for r in doc.recommendations:
                st.markdown(f'<div style="font-size:12px;color:#22c55e;margin-bottom:2px;">✅ {r}</div>', unsafe_allow_html=True)


def render_deepfake_card(det: DeepfakeDetection) -> None:
    sc = stealth_color(det.stealth_level)
    type_colors = {"full_ai": "#ef4444", "parrot": "#f97316", "style_transfer": "#eab308", "template_fill": "#8b5cf6", "hybrid": "#3b82f6"}
    tc = type_colors.get(det.deepfake_type, "#6b7280")
    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.06);border-radius:12px;padding:14px;
         border-left:4px solid {sc};margin-bottom:10px;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
            <span style="font-weight:700;font-size:13px;color:#e2e8f0;">{det.detection_id}</span>
            <div style="display:flex;gap:6px;">
                <span style="font-size:10px;padding:2px 8px;border-radius:8px;background:{tc}20;color:{tc};font-weight:600;">{det.deepfake_type.replace('_', ' ').title()}</span>
                <span style="font-size:10px;padding:2px 8px;border-radius:8px;background:{sc}20;color:{sc};text-transform:uppercase;font-weight:600;">{det.stealth_level} stealth</span>
            </div>
        </div>
        <div style="font-size:12px;color:#cbd5e1;margin-bottom:4px;">📄 {det.doc_title}</div>
        <div style="font-size:11px;color:#94a3b8;margin-bottom:6px;">
            🎯 {det.confidence:.0%} confidence · ⚙️ {det.generation_method} · 🤖 {det.model_fingerprint or 'Unknown'}
        </div>
        <div style="display:flex;gap:4px;flex-wrap:wrap;margin-bottom:4px;">
            {''.join(f'<span style="font-size:9px;padding:2px 6px;border-radius:8px;background:rgba(239,68,68,0.12);color:#fca5a5;">{ind}</span>' for ind in det.indicators[:4])}
        </div>
        <div style="font-size:11px;color:#f59e0b;">⚡ Action: {det.recommended_action}</div>
    </div>
    """, unsafe_allow_html=True)


def render_model_card(model: AIModelFingerprint) -> None:
    acc_color = auth_color(model.detection_accuracy * 100)
    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.06);border-radius:12px;padding:14px;
         border:1px solid rgba(255,255,255,0.08);margin-bottom:10px;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
            <div>
                <span style="font-weight:700;font-size:14px;color:#e2e8f0;">{model.model_name}</span>
                <span style="font-size:11px;color:#94a3b8;margin-left:8px;">{model.model_family} Family</span>
            </div>
            <span style="font-size:14px;font-weight:800;color:{acc_color};">{model.detection_accuracy:.0%}</span>
        </div>
        <div style="display:flex;gap:12px;font-size:11px;color:#94a3b8;margin-bottom:6px;">
            <span>📊 Perplexity: {model.avg_perplexity}</span>
            <span>📈 Burstiness: {model.avg_burstiness}</span>
            <span>🔬 {model.samples_analyzed} samples</span>
        </div>
        <div style="display:flex;gap:4px;flex-wrap:wrap;margin-bottom:6px;">
            {''.join(f'<span style="font-size:9px;padding:2px 6px;border-radius:8px;background:rgba(59,130,246,0.12);color:#93c5fd;">{p}</span>' for p in model.signature_patterns)}
        </div>
        <div style="font-size:10px;color:#94a3b8;">Common tokens: {', '.join(model.typical_vocabulary[:5])}</div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# MAIN DASHBOARD
# =============================================================================


def render_ai_authenticator() -> None:
    st.markdown("""
    <style>
    .block-container { padding-top: 1rem; }
    </style>
    """, unsafe_allow_html=True)

    docs = generate_content_authenticity(25)
    deepfakes = generate_deepfake_detections(docs)
    models = generate_model_fingerprints()
    timeline = generate_timeline(30)

    # Header
    st.markdown("""
    <div style="text-align:center;margin-bottom:20px;">
        <div style="font-size:36px;margin-bottom:8px;">🤖</div>
        <h1 style="font-size:28px;font-weight:800;margin:0;
            background:linear-gradient(135deg,#ef4444,#f59e0b);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            AI Content Authenticator
        </h1>
        <p style="font-size:14px;color:#94a3b8;margin-top:6px;">
            Detect AI-generated content, deepfake text, and synthetic media
        </p>
    </div>
    """, unsafe_allow_html=True)

    # KPIs
    total = len(docs)
    ai_generated = sum(1 for d in docs if d.ai_probability > 0.7)
    ai_assisted = sum(1 for d in docs if 0.4 < d.ai_probability <= 0.7)
    human_written = sum(1 for d in docs if d.ai_probability <= 0.4)
    deepfake_count = len(deepfakes)
    avg_auth = sum(d.authenticity_score for d in docs) / len(docs)
    watermark_count = sum(1 for d in docs if d.provenance.ai_watermark_present)

    cols = st.columns(6)
    kpis = [
        ("Documents", str(total), "analyzed", "#3b82f6"),
        ("AI Generated", str(ai_generated), f"{ai_generated/total:.0%} probability", "#ef4444"),
        ("AI Assisted", str(ai_assisted), f"{ai_assisted/total:.0%} probability", "#f97316"),
        ("Human Written", str(human_written), f"{human_written/total:.0%} probability", "#22c55e"),
        ("Deepfakes", str(deepfake_count), "detected", "#ef4444" if deepfake_count > 3 else "#eab308"),
        ("Watermarks", str(watermark_count), "found", "#8b5cf6"),
    ]
    for col, (l, v, s, c) in zip(cols, kpis):
        with col:
            render_kpi(l, v, s, c)

    # Tabs
    tabs = ["📊 Overview", "🔍 Content", "🎭 Deepfakes", "🤖 Models", "📈 Timeline"]
    selected = st.radio("Tabs", tabs, horizontal=True, label_visibility="collapsed")

    if selected == "📊 Overview":
        _render_overview(docs, models, timeline)
    elif selected == "🔍 Content":
        _render_content(docs)
    elif selected == "🎭 Deepfakes":
        _render_deepfakes(deepfakes, models)
    elif selected == "🤖 Models":
        _render_models(models)
    elif selected == "📈 Timeline":
        _render_timeline(timeline, docs)


def _render_overview(docs, models, timeline):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 📊 Authenticity Distribution")
        ranges = {"0-25% (AI)": 0, "25-50% (Mixed)": 0, "50-75% (Likely Human)": 0, "75-100% (Human)": 0}
        for d in docs:
            if d.authenticity_score < 25: ranges["0-25% (AI)"] += 1
            elif d.authenticity_score < 50: ranges["25-50% (Mixed)"] += 1
            elif d.authenticity_score < 75: ranges["50-75% (Likely Human)"] += 1
            else: ranges["75-100% (Human)"] += 1
        for rng, count in ranges.items():
            pct = count / len(docs)
            st.markdown(f"""
            <div style="margin-bottom:8px;">
                <div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:3px;">
                    <span style="color:#e2e8f0;font-weight:600;">{rng}</span>
                    <span style="color:#94a3b8;">{count} ({pct:.0%})</span>
                </div>
                <div style="height:8px;background:rgba(255,255,255,0.08);border-radius:4px;">
                    <div style="height:100%;width:{pct*100}%;background:{auth_color(float(rng.split('-')[0]))};border-radius:4px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("#### 📊 Provenance Breakdown")
        prov_counts = Counter(d.provenance.origin_method for d in docs)
        prov_colors = {"ai_generated": "#ef4444", "ai_assisted": "#f97316", "hybrid": "#eab308", "human_written": "#22c55e", "unknown": "#94a3b8"}
        for pm, count in prov_counts.most_common():
            pc = prov_colors.get(pm, "#6b7280")
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
                <div style="width:10px;height:10px;border-radius:3px;background:{pc};"></div>
                <span style="font-size:12px;color:#e2e8f0;flex:1;text-transform:capitalize;">{pm.replace('_', ' ')}</span>
                <span style="font-size:12px;font-weight:700;color:{pc};">{count}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("#### 📊 30-Day Detection Trend")
        for t in timeline[-10:]:
            bar_w = (t.ai_detected / t.total_scanned) * 100 if t.total_scanned else 0
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:6px;margin-bottom:2px;">
                <span style="font-size:9px;color:#94a3b8;width:50px;">{t.date[5:]}</span>
                <div style="flex:1;height:10px;background:rgba(255,255,255,0.08);border-radius:3px;">
                    <div style="height:100%;width:{bar_w}%;background:#ef4444;border-radius:3px;"></div>
                </div>
                <span style="font-size:10px;color:#ef4444;font-weight:700;width:24px;">{t.ai_detected}</span>
            </div>
            """, unsafe_allow_html=True)


def _render_content(docs):
    col1, col2, col3 = st.columns(3)
    with col1:
        type_filter = st.selectbox("Content Type", ["All"] + list(set(d.content_type for d in docs)))
    with col2:
        auth_filter = st.selectbox("Authenticity", ["All", "AI Generated (0-25%)", "Mixed (25-50%)", "Likely Human (50-75%)", "Human (75-100%)"])
    with col3:
        sort_by = st.selectbox("Sort By", ["AI Probability", "Authenticity", "Word Count"])

    filtered = docs[:]
    if type_filter != "All":
        filtered = [d for d in filtered if d.content_type == type_filter]
    if auth_filter != "All":
        if "AI Generated" in auth_filter:
            filtered = [d for d in filtered if d.authenticity_score < 25]
        elif "Mixed" in auth_filter:
            filtered = [d for d in filtered if 25 <= d.authenticity_score < 50]
        elif "Likely Human" in auth_filter:
            filtered = [d for d in filtered if 50 <= d.authenticity_score < 75]
        else:
            filtered = [d for d in filtered if d.authenticity_score >= 75]

    if sort_by == "AI Probability":
        filtered.sort(key=lambda d: d.ai_probability, reverse=True)
    elif sort_by == "Authenticity":
        filtered.sort(key=lambda d: d.authenticity_score, reverse=True)
    else:
        filtered.sort(key=lambda d: d.word_count, reverse=True)

    st.markdown(f"**{len(filtered)} documents** matching filters")
    for doc in filtered:
        with st.expander(f"{doc.doc_title} — AI: {doc.ai_probability:.0%}", expanded=False):
            render_auth_card(doc, expanded=True)


def _render_deepfakes(deepfakes, models):
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("#### 🎭 Deepfake Detections")
        for det in deepfakes:
            render_deepfake_card(det)
    with col2:
        st.markdown("#### 📊 Type Breakdown")
        type_counts = Counter(d.deepfake_type for d in deepfakes)
        for dt, count in type_counts.most_common():
            st.markdown(f"• **{dt.replace('_', ' ').title()}**: {count}")
        st.markdown("#### 📊 Stealth Levels")
        stealth_counts = Counter(d.stealth_level for d in deepfakes)
        for sl, count in stealth_counts.most_common():
            sc = stealth_color(sl)
            st.markdown(f'<div style="font-size:12px;color:{sc};margin-bottom:4px;">{sl.title()}: <b>{count}</b></div>', unsafe_allow_html=True)


def _render_models(models):
    st.markdown("#### 🤖 AI Model Fingerprints")
    for model in models:
        render_model_card(model)


def _render_timeline(timeline, docs):
    st.markdown("#### 📈 Detection Timeline")
    max_scanned = max(t.total_scanned for t in timeline) if timeline else 1
    for t in timeline:
        bar_w = (t.total_scanned / max_scanned) * 100 if max_scanned else 0
        ai_w = (t.ai_detected / t.total_scanned) * 100 if t.total_scanned else 0
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:6px;margin-bottom:3px;">
            <span style="font-size:9px;color:#94a3b8;width:50px;">{t.date[5:]}</span>
            <div style="flex:1;height:14px;background:rgba(255,255,255,0.08);border-radius:3px;position:relative;">
                <div style="height:100%;width:{bar_w}%;background:rgba(59,130,246,0.3);border-radius:3px;"></div>
                <div style="height:100%;width:{ai_w}%;background:#ef4444;border-radius:3px;position:absolute;top:0;"></div>
            </div>
            <span style="font-size:10px;color:#94a3b8;width:20px;">{t.total_scanned}</span>
            <span style="font-size:10px;color:#ef4444;font-weight:700;width:20px;">{t.ai_detected}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### 📊 Summary Stats")
    total_scanned = sum(t.total_scanned for t in timeline)
    total_ai = sum(t.ai_detected for t in timeline)
    total_deepfake = sum(t.deepfake_detected for t in timeline)
    avg_conf = sum(t.avg_confidence for t in timeline) / len(timeline) if timeline else 0
    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.04);border-radius:10px;padding:14px;">
        <div style="font-size:12px;color:#cbd5e1;margin-bottom:4px;">📄 Total Scanned: <b>{total_scanned}</b></div>
        <div style="font-size:12px;color:#cbd5e1;margin-bottom:4px;">🤖 AI Detected: <b style="color:#ef4444;">{total_ai}</b> ({total_ai/total_scanned:.0%})</div>
        <div style="font-size:12px;color:#cbd5e1;margin-bottom:4px;">🎭 Deepfakes: <b style="color:#ef4444;">{total_deepfake}</b></div>
        <div style="font-size:12px;color:#cbd5e1;">📊 Avg Confidence: <b style="color:#22c55e;">{avg_conf:.0%}</b></div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# ENTRY POINT
# =============================================================================


def main():
    render_ai_authenticator()


if __name__ == "__main__":
    main()
