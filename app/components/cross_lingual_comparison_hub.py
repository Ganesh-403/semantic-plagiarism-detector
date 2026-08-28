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
Cross-Lingual Comparison Hub
=============================
Enables comparison of documents across different languages with translation
quality analysis, cultural context detection, and cross-lingual plagiarism identification.

Features:
- Multi-language document comparison
- Translation quality scoring
- Cross-lingual semantic similarity
- Cultural context and idiom detection
- Language-specific plagiarism patterns
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
class LanguageProfile:
    """Profile of a document's language characteristics."""

    language: str
    language_code: str
    confidence: float
    word_count: int
    sentence_count: int
    avg_sentence_length: float
    vocabulary_richness: float
    formal_score: float
    technical_density: float
    detected_scripts: List[str]


@dataclass
class TranslationQuality:
    """Quality assessment of a translation."""

    translator: str
    source_lang: str
    target_lang: str
    bleu_score: float
    semantic_preservation: float
    cultural_adaptation: float
    fluency_score: float
    adequacy_score: float
    terminology_consistency: float
    idiom_handling: float
    issues: List[str]
    quality_level: str  # excellent, good, acceptable, poor


@dataclass
class CrossLingualMatch:
    """A cross-lingual similarity match between documents."""

    match_id: str
    source_doc_id: str
    source_lang: str
    target_doc_id: str
    target_lang: str
    similarity_score: float
    match_type: str  # direct_translation, semantic_equivalent, paraphrase_cross, structural_clone
    matched_segments: int
    total_segments: int
    translation_confidence: float
    risk_level: str
    evidence: Dict[str, Any]


@dataclass
class CulturalContext:
    """Cultural context detected in a document."""

    context_id: str
    type: str  # idiom, metaphor, cultural_reference, proverb, humor
    original_text: str
    language: str
    literal_translation: str
    intended_meaning: str
    cultural_origin: str
    adaptation_suggestion: str


@dataclass
class LanguagePairStats:
    """Statistics for a specific language pair comparison."""

    source_lang: str
    target_lang: str
    docs_compared: int
    avg_similarity: float
    plagiarism_rate: float
    avg_translation_quality: float
    common_patterns: List[str]
    risk_factors: List[str]


# =============================================================================
# MOCK DATA
# =============================================================================

LANGUAGES = [
    ("English", "en"),
    ("Spanish", "es"),
    ("French", "fr"),
    ("German", "de"),
    ("Chinese", "zh"),
    ("Japanese", "ja"),
    ("Korean", "ko"),
    ("Portuguese", "pt"),
    ("Arabic", "ar"),
    ("Hindi", "hi"),
    ("Russian", "ru"),
    ("Italian", "it"),
]

DOC_PAIRS = [
    (
        "Research on Neural Machine Translation",
        "Investigación sobre Traducción Automática Neural",
        "en",
        "es",
    ),
    (
        "Deep Learning for Text Analysis",
        "Analyse de Texte par Apprentissage Profond",
        "en",
        "fr",
    ),
    (
        "Semantic Similarity Detection Methods",
        "Methoden zur Erkennung semantischer Ähnlichkeiten",
        "en",
        "de",
    ),
    (
        "Plagiarism Detection in Multilingual Corpora",
        "多语言语料库中的抄袭检测",
        "en",
        "zh",
    ),
    ("Automated Essay Scoring Systems", "自動採点システムの開発", "en", "ja"),
    (
        "Cross-lingual Transfer Learning Survey",
        "Cross-lingual Transfer Learningの調査",
        "en",
        "ja",
    ),
    (
        "Natural Language Processing Advances",
        "Avances en Procesamiento de Lenguaje Natural",
        "en",
        "es",
    ),
    (
        "Information Retrieval in Multiple Languages",
        "Recherche d'Information Multilingue",
        "en",
        "fr",
    ),
    (
        "Text Classification Across Languages",
        "Textklassifizierung über Sprachen hinweg",
        "en",
        "de",
    ),
    ("Machine Translation Quality Estimation", "质量评估机器翻译", "en", "zh"),
    (
        "Sentiment Analysis for Global Content",
        "Análisis de Sentimiento para Contenido Global",
        "en",
        "es",
    ),
    (
        "Document Similarity in Cross-lingual Settings",
        "Similarité de Documents en Contexte Multilingue",
        "en",
        "fr",
    ),
]


def generate_language_profiles() -> List[LanguageProfile]:
    profiles = []
    for name, code in LANGUAGES[:8]:
        profiles.append(
            LanguageProfile(
                language=name,
                language_code=code,
                confidence=round(random.uniform(0.85, 0.99), 3),
                word_count=random.randint(3000, 20000),
                sentence_count=random.randint(100, 800),
                avg_sentence_length=round(random.uniform(12, 28), 1),
                vocabulary_richness=round(random.uniform(0.3, 0.85), 3),
                formal_score=round(random.uniform(0.4, 0.95), 3),
                technical_density=round(random.uniform(0.1, 0.7), 3),
                detected_scripts=(
                    ["Latin", "Cyrillic", "CJK", "Arabic"]
                    if code in ("ru", "zh", "ar", "ja", "ko")
                    else ["Latin"]
                ),
            )
        )
    return profiles


def generate_translations() -> List[TranslationQuality]:
    translators = [
        "Google Translate",
        "DeepL",
        "Microsoft Translator",
        "Custom Model",
        "Human Translator",
    ]
    qualities = []
    for src, tgt, sl, tl in DOC_PAIRS[:10]:
        t = random.choice(translators)
        bleu = random.uniform(25, 95)
        qual = (
            "excellent"
            if bleu > 85
            else "good"
            if bleu > 65
            else "acceptable"
            if bleu > 45
            else "poor"
        )
        issues = []
        if bleu < 60:
            issues.append("Low BLEU score indicates poor translation")
        if random.random() > 0.6:
            issues.append("Terminology inconsistency detected")
        if random.random() > 0.7:
            issues.append("Idiom not properly adapted")
        if random.random() > 0.8:
            issues.append("Cultural reference lost in translation")
        qualities.append(
            TranslationQuality(
                translator=t,
                source_lang=sl,
                target_lang=tl,
                bleu_score=round(bleu, 1),
                semantic_preservation=round(random.uniform(0.5, 0.98), 3),
                cultural_adaptation=round(random.uniform(0.3, 0.95), 3),
                fluency_score=round(random.uniform(0.4, 0.98), 3),
                adequacy_score=round(random.uniform(0.5, 0.97), 3),
                terminology_consistency=round(random.uniform(0.5, 0.99), 3),
                idiom_handling=round(random.uniform(0.2, 0.95), 3),
                issues=issues,
                quality_level=qual,
            )
        )
    return qualities


def generate_cross_lingual_matches() -> List[CrossLingualMatch]:
    match_types = [
        "direct_translation",
        "semantic_equivalent",
        "paraphrase_cross",
        "structural_clone",
    ]
    matches = []
    for i, (src, tgt, sl, tl) in enumerate(DOC_PAIRS):
        sim = random.uniform(0.15, 0.95)
        risk = (
            "critical"
            if sim > 0.85
            else "high"
            if sim > 0.65
            else "medium"
            if sim > 0.4
            else "low"
        )
        matches.append(
            CrossLingualMatch(
                match_id=f"CLM-{i+1:03d}",
                source_doc_id=f"doc_{random.randint(1,100)}",
                source_lang=sl,
                target_doc_id=f"doc_{random.randint(1,100)}",
                target_lang=tl,
                similarity_score=round(sim, 3),
                match_type=random.choice(match_types),
                matched_segments=random.randint(2, 15),
                total_segments=random.randint(10, 30),
                translation_confidence=round(random.uniform(0.4, 0.95), 3),
                risk_level=risk,
                evidence={
                    "exact_phrases": random.randint(1, 20),
                    "structural_similarity": round(random.uniform(0.2, 0.9), 2),
                },
            )
        )
    return matches


def generate_cultural_contexts() -> List[CulturalContext]:
    contexts = [
        CulturalContext(
            "CTX-001",
            "idiom",
            "It's raining cats and dogs",
            "en",
            "Llueven gatos y perros",
            "Heavy rain is falling",
            "English",
            "Replace with regional equivalent",
        ),
        CulturalContext(
            "CTX-002",
            "proverb",
            "The early bird catches the worm",
            "en",
            "El que madruga, Dios le ayuda",
            "Being proactive yields rewards",
            "English",
            "Use local proverb equivalent",
        ),
        CulturalContext(
            "CTX-003",
            "cultural_reference",
            "As American as apple pie",
            "en",
            "Tan español como la paella",
            "Quintessentially American",
            "American",
            "Use culturally relevant local reference",
        ),
        CulturalContext(
            "CTX-004",
            "metaphor",
            "Break a leg",
            "en",
            "¡Mucha mierda!",
            "Good luck before a performance",
            "English",
            "Use theater tradition of the target culture",
        ),
        CulturalContext(
            "CTX-005",
            "idiom",
            "C'est la vie",
            "fr",
            "That's life",
            "Accepting life as it is",
            "French",
            "Direct translation works in this case",
        ),
        CulturalContext(
            "CTX-006",
            "humor",
            "Why do programmers prefer dark mode? Because light attracts bugs",
            "en",
            "¿Por qué a los programadores les gusta el modo oscuro? Porque la luz atrae errores",
            "Programming humor about bugs",
            "English",
            "Adapt joke structure for target language",
        ),
        CulturalContext(
            "CTX-007",
            "cultural_reference",
            "Hit a home run",
            "en",
            "Hacer un golazo",
            "Achieve great success",
            "American Baseball → Soccer",
            "Replace sport metaphor",
        ),
        CulturalContext(
            "CTX-008",
            "proverb",
            "Marco ni marka",
            "es",
            "Beggars can't be choosers",
            "Accept what you get",
            "Spanish",
            "Find equivalent English proverb",
        ),
    ]
    return contexts


def generate_language_pair_stats() -> List[LanguagePairStats]:
    pairs = [
        ("English", "Spanish", 45, 0.42, 0.08, 0.78),
        ("English", "French", 38, 0.38, 0.06, 0.82),
        ("English", "German", 32, 0.35, 0.05, 0.75),
        ("English", "Chinese", 52, 0.55, 0.15, 0.62),
        ("English", "Japanese", 28, 0.48, 0.12, 0.68),
        ("English", "Portuguese", 22, 0.40, 0.07, 0.80),
        ("English", "Korean", 18, 0.50, 0.10, 0.65),
        ("English", "Arabic", 15, 0.52, 0.13, 0.58),
    ]
    stats = []
    for sl, tl, docs, sim, pl, tq in pairs:
        stats.append(
            LanguagePairStats(
                source_lang=sl,
                target_lang=tl,
                docs_compared=docs,
                avg_similarity=round(sim, 3),
                plagiarism_rate=round(pl, 3),
                avg_translation_quality=round(tq, 3),
                common_patterns=[
                    "Literal translation",
                    "False friends",
                    "Register mismatch",
                ],
                risk_factors=[
                    "Different script systems",
                    "Cultural context loss",
                    "Idiom handling",
                ],
            )
        )
    return stats


# =============================================================================
# HELPERS
# =============================================================================


def sim_color(s: float) -> str:
    if s > 0.8:
        return "#ef4444"
    if s > 0.6:
        return "#f97316"
    if s > 0.4:
        return "#eab308"
    return "#22c55e"


def quality_color(q: str) -> str:
    return {
        "excellent": "#22c55e",
        "good": "#3b82f6",
        "acceptable": "#eab308",
        "poor": "#ef4444",
    }.get(q, "#6b7280")


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


def render_match_card(match: CrossLingualMatch) -> None:
    sc = sim_color(match.similarity_score)
    type_colors = {
        "direct_translation": "#ef4444",
        "semantic_equivalent": "#f97316",
        "paraphrase_cross": "#eab308",
        "structural_clone": "#8b5cf6",
    }
    tc = type_colors.get(match.match_type, "#6b7280")
    st.markdown(
        f"""
    <div style="background:rgba(255,255,255,0.06);border-radius:12px;padding:14px;
         border-left:4px solid {sc};margin-bottom:10px;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
            <span style="font-weight:700;font-size:13px;color:#e2e8f0;">{match.match_id}</span>
            <div style="display:flex;gap:6px;">
                <span style="font-size:10px;padding:2px 8px;border-radius:8px;background:{tc}20;color:{tc};text-transform:replace;font-weight:600;">{match.match_type.replace('_', ' ')}</span>
                <span style="font-size:10px;padding:2px 8px;border-radius:8px;background:{sc}20;color:{sc};text-transform:uppercase;font-weight:600;">{match.risk_level}</span>
            </div>
        </div>
        <div style="font-size:12px;color:#cbd5e1;margin-bottom:6px;">
            🌐 {match.source_lang.upper()} → {match.target_lang.upper()} ·
            📊 {match.similarity_score:.0%} similarity ·
            📝 {match.matched_segments}/{match.total_segments} segments ·
            🔄 {match.translation_confidence:.0%} translation confidence
        </div>
        <div style="height:8px;background:rgba(255,255,255,0.08);border-radius:4px;margin-bottom:6px;">
            <div style="height:100%;width:{match.similarity_score*100}%;background:{sc};border-radius:4px;"></div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_translation_card(t: TranslationQuality) -> None:
    qc = quality_color(t.quality_level)
    st.markdown(
        f"""
    <div style="background:rgba(255,255,255,0.06);border-radius:12px;padding:14px;
         border:1px solid rgba(255,255,255,0.08);margin-bottom:10px;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
            <span style="font-weight:700;font-size:13px;color:#e2e8f0;">{t.source_lang.upper()} → {t.target_lang.upper()}</span>
            <div style="display:flex;align-items:center;gap:8px;">
                <span style="font-size:11px;color:#94a3b8;">{t.translator}</span>
                <span style="font-size:10px;padding:2px 10px;border-radius:12px;background:{qc}20;color:{qc};text-transform:uppercase;font-weight:600;">{t.quality_level}</span>
            </div>
        </div>
        <div style="display:grid;grid-template-columns:repeat(3, 1fr);gap:8px;margin-bottom:8px;">
            <div style="text-align:center;"><div style="font-size:18px;font-weight:800;color:{qc};">{t.bleu_score:.0f}</div><div style="font-size:10px;color:#94a3b8;">BLEU Score</div></div>
            <div style="text-align:center;"><div style="font-size:18px;font-weight:800;color:#3b82f6;">{t.semantic_preservation:.0%}</div><div style="font-size:10px;color:#94a3b8;">Semantic</div></div>
            <div style="text-align:center;"><div style="font-size:18px;font-weight:800;color:#8b5cf6;">{t.cultural_adaptation:.0%}</div><div style="font-size:10px;color:#94a3b8;">Cultural</div></div>
        </div>
        <div style="display:flex;gap:12px;font-size:11px;color:#94a3b8;flex-wrap:wrap;">
            <span>💬 Fluency: {t.fluency_score:.0%}</span>
            <span>📋 Adequacy: {t.adequacy_score:.0%}</span>
            <span>📚 Terminology: {t.terminology_consistency:.0%}</span>
            <span>🎭 Idioms: {t.idiom_handling:.0%}</span>
        </div>
        {''.join(f'<div style="font-size:11px;color:#f97316;margin-top:4px;">⚠️ {issue}</div>' for issue in t.issues) if t.issues else '<div style="font-size:11px;color:#22c55e;margin-top:4px;">✅ No issues detected</div>'}
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_cultural_context(ctx: CulturalContext) -> None:
    type_colors = {
        "idiom": "#8b5cf6",
        "proverb": "#22c55e",
        "cultural_reference": "#3b82f6",
        "metaphor": "#f59e0b",
        "humor": "#ec4899",
    }
    tc = type_colors.get(ctx.type, "#6b7280")
    st.markdown(
        f"""
    <div style="background:rgba(255,255,255,0.06);border-radius:12px;padding:14px;
         border-left:3px solid {tc};margin-bottom:10px;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
            <span style="font-size:10px;padding:2px 8px;border-radius:8px;background:{tc}20;color:{tc};text-transform:capitalize;font-weight:600;">{ctx.type}</span>
            <span style="font-size:10px;color:#94a3b8;">🌍 {ctx.cultural_origin}</span>
        </div>
        <div style="font-size:13px;font-weight:600;color:#e2e8f0;margin-bottom:4px;">"{ctx.original_text}"</div>
        <div style="font-size:12px;color:#94a3b8;margin-bottom:4px;">→ "{ctx.literal_translation}"</div>
        <div style="font-size:12px;color:#cbd5e1;margin-bottom:6px;">💡 {ctx.intended_meaning}</div>
        <div style="background:rgba(34,197,94,0.08);border-radius:8px;padding:8px;">
            <div style="font-size:11px;color:#22c55e;">🔧 Adaptation: {ctx.adaptation_suggestion}</div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


# =============================================================================
# MAIN DASHBOARD
# =============================================================================


def render_cross_lingual_hub() -> None:
    st.markdown(
        """
    <style>
    .block-container { padding-top: 1rem; }
    </style>
    """,
        unsafe_allow_html=True,
    )

    profiles = generate_language_profiles()
    translations = generate_translations()
    matches = generate_cross_lingual_matches()
    cultural = generate_cultural_contexts()
    pair_stats = generate_language_pair_stats()

    # Header
    st.markdown(
        """
    <div style="text-align:center;margin-bottom:20px;">
        <div style="font-size:36px;margin-bottom:8px;">🌐</div>
        <h1 style="font-size:28px;font-weight:800;margin:0;
            background:linear-gradient(135deg,#3b82f6,#8b5cf6);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            Cross-Lingual Comparison Hub
        </h1>
        <p style="font-size:14px;color:#94a3b8;margin-top:6px;">
            Compare documents across languages with translation quality and cultural analysis
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # KPIs
    total_comparisons = len(matches)
    high_risk = sum(1 for m in matches if m.risk_level in ("critical", "high"))
    avg_sim = sum(m.similarity_score for m in matches) / len(matches) if matches else 0
    avg_bleu = (
        sum(t.bleu_score for t in translations) / len(translations)
        if translations
        else 0
    )
    cultures_detected = len(cultural)
    langs_covered = len(
        set(m.source_lang for m in matches) | set(m.target_lang for m in matches)
    )

    cols = st.columns(6)
    kpis = [
        ("Comparisons", str(total_comparisons), "cross-lingual pairs", "#3b82f6"),
        (
            "High Risk",
            str(high_risk),
            "plagiarism risk",
            "#ef4444" if high_risk > 3 else "#22c55e",
        ),
        ("Avg Similarity", f"{avg_sim:.0%}", "cross-lingual", sim_color(avg_sim)),
        (
            "Avg BLEU",
            f"{avg_bleu:.0f}",
            "translation quality",
            quality_color("good" if avg_bleu > 65 else "acceptable"),
        ),
        ("Cultural Items", str(cultures_detected), "detected contexts", "#8b5cf6"),
        ("Languages", str(langs_covered), "covered", "#06b6d4"),
    ]
    for col, (l, v, s, c) in zip(cols, kpis):
        with col:
            render_kpi(l, v, s, c)

    # Tabs
    tabs = [
        "📊 Overview",
        "🔄 Comparisons",
        "📝 Translations",
        "🎭 Cultural",
        "📈 Language Pairs",
    ]
    selected = st.radio("Tabs", tabs, horizontal=True, label_visibility="collapsed")

    if selected == "📊 Overview":
        _render_overview(profiles, matches, translations, pair_stats)
    elif selected == "🔄 Comparisons":
        _render_comparisons(matches)
    elif selected == "📝 Translations":
        _render_translations(translations)
    elif selected == "🎭 Cultural":
        _render_cultural(cultural, profiles)
    elif selected == "📈 Language Pairs":
        _render_language_pairs(pair_stats, profiles)


def _render_overview(profiles, matches, translations, pair_stats):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🌐 Language Profiles")
        for p in profiles:
            st.markdown(
                f"""
            <div style="background:rgba(255,255,255,0.04);border-radius:10px;padding:10px;margin-bottom:6px;">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
                    <span style="font-weight:700;font-size:13px;color:#e2e8f0;">{p.language} ({p.language_code.upper()})</span>
                    <span style="font-size:11px;color:#22c55e;">{p.confidence:.0%} detected</span>
                </div>
                <div style="display:flex;gap:12px;font-size:10px;color:#94a3b8;">
                    <span>📄 {p.word_count:,} words</span>
                    <span>📝 {p.sentence_count} sentences</span>
                    <span>📚 {p.vocabulary_richness:.0%} richness</span>
                    <span>📋 {p.formal_score:.0%} formal</span>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    with col2:
        st.markdown("#### 📊 Similarity Distribution")
        sim_ranges = {"0-20%": 0, "20-40%": 0, "40-60%": 0, "60-80%": 0, "80-100%": 0}
        for m in matches:
            if m.similarity_score < 0.2:
                sim_ranges["0-20%"] += 1
            elif m.similarity_score < 0.4:
                sim_ranges["20-40%"] += 1
            elif m.similarity_score < 0.6:
                sim_ranges["40-60%"] += 1
            elif m.similarity_score < 0.8:
                sim_ranges["60-80%"] += 1
            else:
                sim_ranges["80-100%"] += 1
        for rng, count in sim_ranges.items():
            pct = count / len(matches) if matches else 0
            st.markdown(
                f"""
            <div style="margin-bottom:6px;">
                <div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:3px;">
                    <span style="color:#e2e8f0;font-weight:600;">{rng}</span>
                    <span style="color:#94a3b8;">{count} ({pct:.0%})</span>
                </div>
                <div style="height:8px;background:rgba(255,255,255,0.08);border-radius:4px;">
                    <div style="height:100%;width:{pct*100}%;background:{sim_color(0.1 + float(rng.split('-')[0].replace('%',''))/100)};border-radius:4px;"></div>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        st.markdown("#### 📊 Top Language Pairs by Volume")
        for ps in sorted(pair_stats, key=lambda x: x.docs_compared, reverse=True)[:5]:
            st.markdown(
                f"• **{ps.source_lang} → {ps.target_lang}**: {ps.docs_compared} docs, {ps.avg_similarity:.0%} avg sim"
            )


def _render_comparisons(matches):
    col1, col2, col3 = st.columns(3)
    with col1:
        risk_filter = st.selectbox(
            "Risk Level", ["All", "Critical", "High", "Medium", "Low"]
        )
    with col2:
        type_filter = st.selectbox(
            "Match Type", ["All"] + list(set(m.match_type for m in matches))
        )
    with col3:
        sort_by = st.selectbox(
            "Sort By", ["Similarity (High→Low)", "Similarity (Low→High)", "Segments"]
        )

    filtered = matches[:]
    if risk_filter != "All":
        filtered = [m for m in filtered if m.risk_level == risk_filter.lower()]
    if type_filter != "All":
        filtered = [m for m in filtered if m.match_type == type_filter]

    if sort_by == "Similarity (High→Low)":
        filtered.sort(key=lambda m: m.similarity_score, reverse=True)
    elif sort_by == "Segments":
        filtered.sort(key=lambda m: m.matched_segments, reverse=True)
    else:
        filtered.sort(key=lambda m: m.similarity_score)

    st.markdown(f"**{len(filtered)} matches** found")
    for m in filtered:
        render_match_card(m)


def _render_translations(translations):
    st.markdown("#### 📝 Translation Quality Analysis")
    for t in translations:
        render_translation_card(t)


def _render_cultural(cultural, profiles):
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("#### 🎭 Cultural Context Detection")
        for ctx in cultural:
            render_cultural_context(ctx)
    with col2:
        st.markdown("#### 📊 Context Type Distribution")
        type_counts = Counter(c.type for c in cultural)
        for t, count in type_counts.most_common():
            st.markdown(f"• **{t.replace('_', ' ').title()}**: {count}")
        st.markdown("#### 📊 Cultural Origins")
        origin_counts = Counter(c.cultural_origin for c in cultural)
        for o, count in origin_counts.most_common():
            st.markdown(f"• **{o}**: {count}")


def _render_language_pairs(pair_stats, profiles):
    st.markdown("#### 📈 Language Pair Analysis")
    for ps in pair_stats:
        rc = sim_color(ps.avg_similarity)
        pqc = quality_color(
            "good" if ps.avg_translation_quality > 0.7 else "acceptable"
        )
        st.markdown(
            f"""
        <div style="background:rgba(255,255,255,0.06);border-radius:12px;padding:14px;
             border:1px solid rgba(255,255,255,0.08);margin-bottom:10px;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                <span style="font-weight:700;font-size:14px;color:#e2e8f0;">{ps.source_lang} → {ps.target_lang}</span>
                <span style="font-size:12px;color:#94a3b8;">{ps.docs_compared} docs compared</span>
            </div>
            <div style="display:flex;gap:16px;margin-bottom:6px;">
                <span style="font-size:12px;color:#94a3b8;">📊 Similarity: <b style="color:{rc}">{ps.avg_similarity:.0%}</b></span>
                <span style="font-size:12px;color:#94a3b8;">⚠️ Plagiarism: <b style="color:{sim_color(ps.plagiarism_rate * 2)}">{ps.plagiarism_rate:.0%}</b></span>
                <span style="font-size:12px;color:#94a3b8;">📝 Translation: <b style="color:{pqc}">{ps.avg_translation_quality:.0%}</b></span>
            </div>
            <div style="display:flex;gap:4px;flex-wrap:wrap;">
                {''.join(f'<span style="font-size:9px;padding:2px 6px;border-radius:8px;background:rgba(239,68,68,0.12);color:#fca5a5;">{rf}</span>' for rf in ps.risk_factors[:3])}
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )


# =============================================================================
# ENTRY POINT
# =============================================================================


def main():
    render_cross_lingual_hub()


if __name__ == "__main__":
    main()
