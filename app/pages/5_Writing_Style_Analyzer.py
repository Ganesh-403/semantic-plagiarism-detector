"""
app/pages/5_Writing_Style_Analyzer.py
-------------------------------------
Streamlit multi-page app: Writing Style Analyzer & Authorship Attribution.

Analyzes writing style fingerprints including vocabulary richness, sentence
complexity, readability scores, punctuation patterns, and structural habits
for authorship attribution and style comparison.
"""

import math
import re
from collections import Counter
from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Writing Style Analyzer - Plagiarism Detector",
    page_icon="✍️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Style analysis engine
# ---------------------------------------------------------------------------

def _compute_style_profile(text: str, author: str = "Unknown") -> dict[str, Any]:
    """Compute a comprehensive writing style profile from text."""
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    words = text.split()
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

    # Word length stats
    word_lengths = [len(w) for w in words]
    avg_word_len = sum(word_lengths) / len(word_lengths) if word_lengths else 0

    # Sentence length stats
    sent_lengths = [len(s.split()) for s in sentences]
    avg_sent_len = sum(sent_lengths) / len(sent_lengths) if sent_lengths else 0

    # Vocabulary richness
    unique_words = set(w.lower() for w in words if w.isalpha())
    total_alpha_words = [w for w in words if w.isalpha()]
    vocab_richness = len(unique_words) / len(total_alpha_words) if total_alpha_words else 0

    # Hapax legomena (words used only once)
    word_freq = Counter(w.lower() for w in total_alpha_words)
    hapax = sum(1 for v in word_freq.values() if v == 1)
    hapax_ratio = hapax / len(total_alpha_words) if total_alpha_words else 0

    # Punctuation patterns
    commas = text.count(',')
    semicolons = text.count(';')
    colons = text.count(':')
    dashes = text.count('—') + text.count('--')
    exclamation = text.count('!')
    question = text.count('?')
    parens = text.count('(') + text.count(')')
    quotes = text.count('"') + text.count("'")

    # Readability (Flesch-Kincaid approximation)
    syllable_count = sum(_count_syllables(w) for w in total_alpha_words)
    fk_grade = (0.39 * avg_sent_len) + (11.8 * (syllable_count / len(total_alpha_words) if total_alpha_words else 0)) - 15.59

    # Flesch Reading Ease
    if avg_sent_len > 0 and total_alpha_words:
        fre = 206.835 - (1.015 * avg_sent_len) - (84.6 * (syllable_count / len(total_alpha_words)))
    else:
        fre = 0

    # Type-Token Ratio (sliding window)
    ttr_values = []
    window = 50
    for i in range(0, max(len(total_alpha_words) - window, 1), window):
        chunk = total_alpha_words[i:i + window]
        ttr = len(set(w.lower() for w in chunk)) / len(chunk) if chunk else 0
        ttr_values.append(round(ttr, 3))
    avg_ttr = sum(ttr_values) / len(ttr_values) if ttr_values else 0

    # Sentence starters
    starters = Counter()
    for s in sentences:
        first = s.split()[0].lower() if s.split() else ""
        starters[first] += 1

    # Pronoun usage
    pronouns_i = sum(1 for w in words if w.lower() in ('i', 'me', 'my', 'mine', 'myself'))
    pronouns_you = sum(1 for w in words if w.lower() in ('you', 'your', 'yours', 'yourself'))
    pronouns_third = sum(1 for w in words if w.lower() in ('he', 'she', 'it', 'they', 'him', 'her', 'his', 'their'))
    pronouns_total = pronouns_i + pronouns_you + pronouns_third

    # Passive voice indicators
    be_forms = ['is', 'are', 'was', 'were', 'be', 'been', 'being']
    past_participle_count = sum(
        1
        for i, w in enumerate(words)
        if w.lower() in be_forms
        and i + 1 < len(words)
        and words[i + 1]
        and words[i + 1][0].isupper()
    )

    # Adjective/adverb density
    ly_words = sum(1 for w in words if w.lower().endswith('ly') and len(w) > 3)

    # Complex word percentage (3+ syllables)
    complex_words = sum(1 for w in total_alpha_words if _count_syllables(w) >= 3)
    complex_pct = (complex_words / len(total_alpha_words) * 100) if total_alpha_words else 0

    return {
        "author": author,
        "word_count": len(words),
        "sentence_count": len(sentences),
        "paragraph_count": len(paragraphs),
        "unique_words": len(unique_words),
        "avg_word_length": round(avg_word_len, 2),
        "avg_sentence_length": round(avg_sent_len, 2),
        "vocab_richness": round(vocab_richness, 3),
        "hapax_ratio": round(hapax_ratio, 3),
        "fk_grade": round(max(fk_grade, 0), 1),
        "fre_score": round(max(min(fre, 100), 0), 1),
        "avg_ttr": round(avg_ttr, 3),
        "comma_rate": round(commas / len(sentences), 2) if sentences else 0,
        "semicolon_rate": round(semicolons / len(sentences), 2) if sentences else 0,
        "colon_rate": round(colons / len(sentences), 2) if sentences else 0,
        "dash_rate": round(dashes / len(sentences), 2) if sentences else 0,
        "exclamation_rate": round(exclamation / len(sentences), 2) if sentences else 0,
        "question_rate": round(question / len(sentences), 2) if sentences else 0,
        "paren_rate": round(parens / len(sentences), 2) if sentences else 0,
        "pronoun_i_ratio": round(pronouns_i / len(words) * 100, 2) if words else 0,
        "pronoun_you_ratio": round(pronouns_you / len(words) * 100, 2) if words else 0,
        "pronoun_third_ratio": round(pronouns_third / len(words) * 100, 2) if words else 0,
        "passive_indicators": past_participle_count,
        "adverb_density": round(ly_words / len(words) * 100, 2) if words else 0,
        "complex_word_pct": round(complex_pct, 1),
        "syllable_count": syllable_count,
        "hapax_count": hapax,
        "top_words": word_freq.most_common(15),
        "top_starters": starters.most_common(8),
    }


def _count_syllables(word: str) -> int:
    """Rough syllable count for English words."""
    word = word.lower().strip()
    if len(word) <= 3:
        return 1
    vowels = 'aeiou'
    count = 0
    prev_vowel = False
    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    if word.endswith('e') and count > 1:
        count -= 1
    if word.endswith('le') and len(word) > 2 and word[-3] not in vowels:
        count += 1
    return max(count, 1)


def _compute_similarity(profile_a: dict, profile_b: dict) -> float:
    """Compute style similarity between two author profiles (0-100)."""
    numeric_keys = [
        'avg_word_length', 'avg_sentence_length', 'vocab_richness',
        'hapax_ratio', 'fk_grade', 'fre_score', 'avg_ttr',
        'comma_rate', 'semicolon_rate', 'colon_rate', 'dash_rate',
        'exclamation_rate', 'question_rate', 'paren_rate',
        'pronoun_i_ratio', 'pronoun_you_ratio', 'pronoun_third_ratio',
        'adverb_density', 'complex_word_pct',
    ]
    diffs = []
    ranges = {
        'avg_word_length': (3, 7),
        'avg_sentence_length': (5, 40),
        'vocab_richness': (0.1, 0.9),
        'hapax_ratio': (0.1, 0.9),
        'fk_grade': (1, 20),
        'fre_score': (0, 100),
        'avg_ttr': (0.3, 0.9),
        'comma_rate': (0, 4),
        'semicolon_rate': (0, 1.5),
        'colon_rate': (0, 1),
        'dash_rate': (0, 1),
        'exclamation_rate': (0, 1),
        'question_rate': (0, 1),
        'paren_rate': (0, 1),
        'pronoun_i_ratio': (0, 5),
        'pronoun_you_ratio': (0, 5),
        'pronoun_third_ratio': (0, 10),
        'adverb_density': (0, 8),
        'complex_word_pct': (0, 40),
    }
    for k in numeric_keys:
        a_val = profile_a.get(k, 0)
        b_val = profile_b.get(k, 0)
        lo, hi = ranges.get(k, (0, max(a_val, b_val, 1)))
        norm_range = hi - lo if hi != lo else 1
        diff = abs(a_val - b_val) / norm_range
        diffs.append(diff)
    avg_diff = sum(diffs) / len(diffs) if diffs else 0
    return round((1 - avg_diff) * 100, 1)


# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------

def _generate_mock_documents() -> list[dict[str, Any]]:
    """Generate mock documents with style profiles."""
    import random
    random.seed(42)
    authors = [
        "Alice Johnson", "Bob Smith", "Carol White", "David Brown", "Eva Martinez",
    ]
    doc_titles = [
        "AI Ethics in Modern Society", "Climate Change Impacts on Coastal Cities",
        "Quantum Computing Fundamentals", "Machine Learning in Healthcare",
        "Sustainable Energy Solutions", "Digital Privacy and Security",
        "Cognitive Science and AI", "Bioethics in Genetic Engineering",
        "Urban Planning for Smart Cities", "Renewable Materials Research",
        "Cybersecurity Threat Landscape", "Neural Networks Deep Dive",
        "Renewable Energy Policy Analysis", "Data Privacy Regulations",
        "Artificial Intelligence in Education",
    ]
    departments = [
        "Computer Science", "Philosophy", "Environmental Science",
        "Physics", "Medicine", "Law", "Psychology",
    ]
    docs = []
    for i, title in enumerate(doc_titles):
        author = authors[i % len(authors)]
        dept = departments[i % len(departments)]
        sim_to_author = random.uniform(0.6, 0.95)
        base_word_count = random.randint(1500, 8000)
        docs.append({
            "id": f"DOC-{200 + i}",
            "title": title,
            "author": author,
            "department": dept,
            "word_count": base_word_count,
            "uploaded": (datetime.now() - random.randint(1, 60) * __import__('datetime').timedelta(days=1)).strftime("%Y-%m-%d"),
            "similarity_to_author": round(sim_to_author * 100, 1),
            "flagged": sim_to_author < 0.75,
        })
    return docs


def _generate_mock_style_profiles() -> list[dict[str, Any]]:
    """Generate mock pre-computed style profiles for each author."""
    import random
    random.seed(42)
    profiles = []
    author_bases = {
        "Alice Johnson": {"avg_word_length": 5.8, "avg_sentence_length": 22, "vocab_richness": 0.62, "fk_grade": 12.5, "fre_score": 48, "pronoun_i_ratio": 2.1, "adverb_density": 3.2, "complex_word_pct": 18},
        "Bob Smith": {"avg_word_length": 4.9, "avg_sentence_length": 14, "vocab_richness": 0.55, "fk_grade": 8.2, "fre_score": 68, "pronoun_i_ratio": 3.8, "adverb_density": 1.8, "complex_word_pct": 10},
        "Carol White": {"avg_word_length": 6.1, "avg_sentence_length": 28, "vocab_richness": 0.71, "fk_grade": 15.8, "fre_score": 35, "pronoun_i_ratio": 1.2, "adverb_density": 4.5, "complex_word_pct": 24},
        "David Brown": {"avg_word_length": 5.2, "avg_sentence_length": 18, "vocab_richness": 0.58, "fk_grade": 10.1, "fre_score": 55, "pronoun_i_ratio": 2.8, "adverb_density": 2.5, "complex_word_pct": 14},
        "Eva Martinez": {"avg_word_length": 5.5, "avg_sentence_length": 20, "vocab_richness": 0.65, "fk_grade": 11.8, "fre_score": 52, "pronoun_i_ratio": 1.9, "adverb_density": 3.8, "complex_word_pct": 16},
    }
    for author, base in author_bases.items():
        profile = {k: v for k, v in base.items()}
        profile["author"] = author
        profile["word_count"] = random.randint(5000, 15000)
        profile["sentence_count"] = random.randint(200, 800)
        profile["unique_words"] = int(profile["word_count"] * profile["vocab_richness"])
        profile["hapax_ratio"] = round(random.uniform(0.3, 0.55), 3)
        profile["avg_ttr"] = round(profile["vocab_richness"] * random.uniform(0.9, 1.1), 3)
        profile["comma_rate"] = round(random.uniform(0.8, 2.5), 2)
        profile["semicolon_rate"] = round(random.uniform(0.05, 0.4), 2)
        profile["colon_rate"] = round(random.uniform(0.05, 0.3), 2)
        profile["dash_rate"] = round(random.uniform(0.02, 0.15), 2)
        profile["exclamation_rate"] = round(random.uniform(0, 0.1), 2)
        profile["question_rate"] = round(random.uniform(0.05, 0.3), 2)
        profile["paren_rate"] = round(random.uniform(0, 0.2), 2)
        profile["pronoun_you_ratio"] = round(random.uniform(0.5, 2.0), 2)
        profile["pronoun_third_ratio"] = round(random.uniform(3, 8), 2)
        profile["passive_indicators"] = random.randint(10, 80)
        profile["syllable_count"] = int(profile["word_count"] * 1.6)
        profile["hapax_count"] = int(profile["word_count"] * profile["hapax_ratio"])
        profile["top_words"] = [
            ("the", random.randint(80, 200)), ("of", random.randint(40, 100)),
            ("and", random.randint(30, 80)), ("to", random.randint(25, 70)),
            ("in", random.randint(20, 60)), ("is", random.randint(15, 50)),
            ("for", random.randint(10, 40)), ("that", random.randint(8, 35)),
            ("with", random.randint(6, 25)), ("on", random.randint(5, 20)),
            ("this", random.randint(5, 18)), ("are", random.randint(4, 15)),
            ("as", random.randint(4, 12)), ("by", random.randint(3, 10)),
            ("from", random.randint(3, 10)),
        ]
        profile["top_starters"] = [
            ("the", random.randint(20, 60)), ("this", random.randint(10, 30)),
            ("in", random.randint(8, 25)), ("a", random.randint(6, 20)),
            ("it", random.randint(5, 15)), ("we", random.randint(3, 12)),
            ("they", random.randint(2, 8)), ("however", random.randint(2, 10)),
        ]
        profiles.append(profile)
    return profiles


# ---------------------------------------------------------------------------
# Chart renderers
# ---------------------------------------------------------------------------

def _render_radar_comparison(profiles: list[dict], metrics: list[str], labels: list[str]):
    """Render a side-by-side metric comparison as grouped horizontal bars."""
    colors = ["#4a90d9", "#e83e8c", "#28a745", "#fd7e14", "#6f42c1"]
    st.markdown("**Style Metric Comparison:**")
    for metric, label in zip(metrics, labels):
        values = []
        for i, p in enumerate(profiles):
            values.append((p.get("author", f"Author {i+1}"), p.get(metric, 0), colors[i % len(colors)]))
        mx = max((abs(v) for _, v, _ in values), default=1) or 1
        bars_html = ""
        for author, val, color in values:
            pct = abs(val) / mx * 100 if mx else 0
            bars_html += (
                f'<div style="margin:2px 0;display:flex;align-items:center">'
                f'<span style="width:110px;font-size:0.78em;text-align:right;margin-right:6px">{author[:14]}</span>'
                f'<div style="width:60%;background:#e8e8e8;border-radius:3px;height:14px">'
                f'<div style="width:{pct:.0f}%;background:{color};border-radius:3px;height:100%"></div></div>'
                f'<span style="margin-left:6px;font-size:0.78em;font-weight:600">{val}</span></div>'
            )
        st.markdown(f"**{label}**", help=f"Metric: {metric}")
        st.markdown(bars_html, unsafe_allow_html=True)


def _render_word_cloud_simple(top_words: list[tuple[str, int]], title: str, color: str = "#4a90d9"):
    """Render a simple word frequency display as sized inline elements."""
    if not top_words:
        return
    max_count = top_words[0][1] if top_words else 1
    words_html = ""
    for word, count in top_words[:12]:
        size = 0.7 + (count / max_count) * 0.9
        opacity = 0.5 + (count / max_count) * 0.5
        words_html += (
            f'<span style="display:inline-block;margin:3px 5px;padding:4px 10px;'
            f'background:{color};color:white;border-radius:14px;font-size:{size:.2f}em;'
            f'opacity:{opacity:.2f}">{word} ({count})</span>'
        )
    st.markdown(f"**{title}**")
    st.markdown(words_html, unsafe_allow_html=True)


def _render_profile_card(profile: dict, highlight: bool = False):
    """Render an author style profile card."""
    border = "3px solid #4a90d9" if highlight else "1px solid #ddd"
    bg = "#f0f7ff" if highlight else "#fafafa"
    st.markdown(
        f'<div style="border:{border};border-radius:8px;padding:14px;margin:8px 0;background:{bg}">'
        f'<h4 style="margin:0 0 8px 0">👤 {profile["author"]}</h4>'
        f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;font-size:0.88em">'
        f'<div>📝 Words: <strong>{profile["word_count"]:,}</strong></div>'
        f'<div>📖 Sentences: <strong>{profile["sentence_count"]:,}</strong></div>'
        f'<div>🔤 Unique: <strong>{profile["unique_words"]:,}</strong></div>'
        f'<div>📏 Avg Word Len: <strong>{profile["avg_word_length"]}</strong></div>'
        f'<div>📐 Avg Sent Len: <strong>{profile["avg_sentence_length"]}</strong></div>'
        f'<div>📚 Vocab Richness: <strong>{profile["vocab_richness"]}</strong></div>'
        f'<div>🎯 FK Grade: <strong>{profile["fk_grade"]}</strong></div>'
        f'<div>✨ Readability: <strong>{profile["fre_score"]}</strong></div>'
        f'<div>🧩 Hapax Ratio: <strong>{profile["hapax_ratio"]}</strong></div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )


def _render_similarity_gauge(score: float, label: str = "Style Match"):
    """Render a circular similarity gauge using SVG."""
    angle = (score / 100) * 360
    rad = math.pi * angle / 180
    r = 45
    cx, cy = 60, 60
    x_end = cx + r * math.sin(rad)
    y_end = cy - r * math.cos(rad)
    large_arc = 1 if angle > 180 else 0
    color = "#dc3545" if score < 50 else "#ffc107" if score < 70 else "#28a745"

    svg = f'''<svg width="120" height="130" viewBox="0 0 120 130" xmlns="http://www.w3.org/2000/svg">
  <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="#e0e0e0" stroke-width="8"/>
  <path d="M {cx} {cy - r} A {r} {r} 0 {large_arc} 1 {x_end:.1f} {y_end:.1f}"
        fill="none" stroke="{color}" stroke-width="8" stroke-linecap="round"/>
  <text x="{cx}" y="{cy + 5}" text-anchor="middle" font-size="18" font-weight="bold" fill="{color}">{score:.0f}%</text>
  <text x="{cx}" y="{cy + 22}" text-anchor="middle" font-size="10" fill="#666">{label}</text>
</svg>'''
    st.markdown(svg, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------

def _render_overview(profiles: list[dict]):
    """Render the overview comparison of all author profiles."""
    st.subheader("📊 Author Style Overview")

    cols = st.columns(min(len(profiles), 5))
    for idx, p in enumerate(profiles):
        with cols[idx % len(cols)]:
            _render_profile_card(p, highlight=(idx == 0))

    # Summary table
    st.markdown("**Summary Comparison Table:**")
    rows = []
    for p in profiles:
        rows.append({
            "Author": p["author"],
            "Words": p["word_count"],
            "Sentences": p["sentence_count"],
            "Avg Word": p["avg_word_length"],
            "Avg Sent": p["avg_sentence_length"],
            "Vocab Rich": p["vocab_richness"],
            "FK Grade": p["fk_grade"],
            "Readability": p["fre_score"],
            "Complex %": p["complex_word_pct"],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_vocabulary_analysis(profiles: list[dict]):
    """Render vocabulary richness and word usage analysis."""
    st.subheader("📚 Vocabulary Analysis")

    metrics = ['vocab_richness', 'hapax_ratio', 'avg_ttr', 'complex_word_pct', 'avg_word_length']
    labels = ['Vocabulary Richness', 'Hapax Legomena Ratio', 'Type-Token Ratio', 'Complex Word %', 'Avg Word Length']
    _render_radar_comparison(profiles, metrics, labels)

    st.markdown("---")
    for p in profiles:
        _render_word_cloud_simple(
            p.get("top_words", []),
            f"Top Words — {p['author']}",
            color="#4a90d9",
        )
        st.markdown("")


def _render_sentence_analysis(profiles: list[dict]):
    """Render sentence structure and complexity analysis."""
    st.subheader("📐 Sentence Structure")

    metrics = ['avg_sentence_length', 'fk_grade', 'fre_score', 'adverb_density', 'passive_indicators']
    labels = ['Avg Sentence Length', 'Flesch-Kincaid Grade', 'Reading Ease Score', 'Adverb Density (%)', 'Passive Voice Count']
    _render_radar_comparison(profiles, metrics, labels)

    st.markdown("---")
    # Readability scale
    st.markdown("**Readability Scale:**")
    readability_data = {}
    for p in profiles:
        readability_data[p["author"]] = p["fre_score"]

    scale_labels = [
        (90, 100, "Very Easy (5th grade)"),
        (80, 89, "Easy (6th grade)"),
        (70, 79, "Fairly Easy (7th grade)"),
        (60, 69, "Standard (8th-9th grade)"),
        (50, 59, "Fairly Difficult (10th-12th)"),
        (30, 49, "Difficult (College)"),
        (0, 29, "Very Difficult (Graduate)"),
    ]

    for author, score in sorted(readability_data.items(), key=lambda x: x[1], reverse=True):
        level = next((l for lo, hi, l in scale_labels if lo <= score <= hi), "Unknown")
        color = "#28a745" if score >= 60 else "#ffc107" if score >= 40 else "#dc3545"
        st.markdown(
            f'<div style="display:flex;align-items:center;margin:4px 0">'
            f'<span style="width:130px;font-size:0.88em;font-weight:600">{author}</span>'
            f'<div style="width:50%;background:#e0e0e0;border-radius:4px;height:16px">'
            f'<div style="width:{score}%;background:{color};border-radius:4px;height:100%"></div></div>'
            f'<span style="margin-left:8px;font-size:0.85em">{score} — {level}</span></div>',
            unsafe_allow_html=True,
        )


def _render_punctuation_analysis(profiles: list[dict]):
    """Render punctuation usage patterns."""
    st.subheader("✏️ Punctuation Patterns")

    metrics = ['comma_rate', 'semicolon_rate', 'colon_rate', 'dash_rate', 'exclamation_rate', 'question_rate', 'paren_rate']
    labels = ['Commas/Sentence', 'Semicolons/Sentence', 'Colons/Sentence', 'Dashes/Sentence', 'Exclamations/Sentence', 'Questions/Sentence', 'Parentheses/Sentence']
    _render_radar_comparison(profiles, metrics, labels)

    # Punctuation fingerprint
    st.markdown("**Punctuation Fingerprint:**")
    rows = []
    for p in profiles:
        rows.append({
            "Author": p["author"],
            "Comma": p["comma_rate"],
            "Semicolon": p["semicolon_rate"],
            "Colon": p["colon_rate"],
            "Dash": p["dash_rate"],
            "Exclamation": p["exclamation_rate"],
            "Question": p["question_rate"],
            "Parenthesis": p["paren_rate"],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_pronoun_analysis(profiles: list[dict]):
    """Render pronoun and perspective analysis."""
    st.subheader("👤 Pronoun & Perspective")

    metrics = ['pronoun_i_ratio', 'pronoun_you_ratio', 'pronoun_third_ratio']
    labels = ['First Person (I/me/my)', 'Second Person (you/your)', 'Third Person (he/she/they)']
    _render_radar_comparison(profiles, metrics, labels)

    # Perspective classification
    st.markdown("**Perspective Classification:**")
    for p in profiles:
        i_pct = p["pronoun_i_ratio"]
        you_pct = p["pronoun_you_ratio"]
        third_pct = p["pronoun_third_ratio"]
        if i_pct > you_pct and i_pct > third_pct:
            perspective = "First Person (Academic/Personal)"
            icon = "🙋"
            color = "#4a90d9"
        elif you_pct > third_pct:
            perspective = "Second Person (Direct/Instructional)"
            icon = "👉"
            color = "#e83e8c"
        else:
            perspective = "Third Person (Formal/Scientific)"
            icon = "🔬"
            color = "#28a745"
        st.markdown(
            f'<div style="border-left:4px solid {color};padding:8px 12px;margin:6px 0;background:#f8f9fa;border-radius:4px">'
            f'{icon} <strong>{p["author"]}</strong> — {perspective}<br/>'
            f'<span style="font-size:0.82em;color:#666">I: {i_pct}% | You: {you_pct}% | Third: {third_pct}%</span></div>',
            unsafe_allow_html=True,
        )


def _render_authorship_comparison(profiles: list[dict]):
    """Render side-by-side authorship comparison."""
    st.subheader("🔍 Authorship Comparison")

    if len(profiles) < 2:
        st.info("Need at least 2 authors for comparison.")
        return

    # Pairwise similarity matrix
    st.markdown("**Style Similarity Matrix:**")
    n = len(profiles)
    matrix_data = []
    for i in range(n):
        row = {"Author": profiles[i]["author"]}
        for j in range(n):
            if i == j:
                row[profiles[j]["author"][:12]] = "—"
            else:
                sim = _compute_similarity(profiles[i], profiles[j])
                row[profiles[j]["author"][:12]] = f"{sim}%"
        matrix_data.append(row)
    st.dataframe(pd.DataFrame(matrix_data), use_container_width=True, hide_index=True)

    # Most similar pairs
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = _compute_similarity(profiles[i], profiles[j])
            pairs.append((profiles[i]["author"], profiles[j]["author"], sim))
    pairs.sort(key=lambda x: x[2], reverse=True)

    st.markdown("**Most Similar Author Pairs:**")
    for a1, a2, sim in pairs:
        color = "#28a745" if sim >= 70 else "#ffc107" if sim >= 50 else "#fd7e14"
        st.markdown(
            f'<div style="display:flex;align-items:center;margin:4px 0;padding:6px;background:#f8f9fa;border-radius:4px">'
            f'<span style="width:120px;font-size:0.88em;font-weight:600">{a1}</span>'
            f'<span style="margin:0 8px">⟷</span>'
            f'<span style="width:120px;font-size:0.88em;font-weight:600">{a2}</span>'
            f'<span style="margin-left:auto;color:{color};font-weight:700;font-size:1.1em">{sim}%</span></div>',
            unsafe_allow_html=True,
        )

    # Side-by-side deep comparison
    if len(profiles) >= 2:
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            _render_profile_card(profiles[0], highlight=True)
        with c2:
            _render_profile_card(profiles[1], highlight=False)


def _render_forensic_flags(profiles: list[dict]):
    """Render authorship anomaly detection and forensic flags."""
    st.subheader("🚨 Authorship Anomaly Detection")

    # Flag suspicious patterns
    flags = []
    for p in profiles:
        author_flags = []
        if p["avg_sentence_length"] > 30:
            author_flags.append(("Unusually long sentences", "high", "May indicate mechanical writing"))
        if p["vocab_richness"] < 0.45:
            author_flags.append(("Low vocabulary diversity", "high", "Repetitive word usage"))
        if p["complex_word_pct"] > 25:
            author_flags.append(("High complexity words", "medium", "Possible copy from academic source"))
        if p["fre_score"] < 30:
            author_flags.append(("Very low readability", "medium", "Dense academic text"))
        if p["adverb_density"] > 5:
            author_flags.append(("Excessive adverbs", "low", "Stylistic preference"))
        if p["passive_indicators"] > 60:
            author_flags.append(("High passive voice", "medium", "May indicate academic plagiarism"))
        if p["avg_word_length"] > 6.5:
            author_flags.append(("Long word average", "low", "Technical vocabulary"))
        if p["exclamation_rate"] > 0.08:
            author_flags.append(("Unusual exclamation usage", "low", "Informal writing style"))
        if author_flags:
            flags.append((p["author"], author_flags))

    if not flags:
        st.success("✅ No style anomalies detected across all profiles.")
    else:
        for author, author_flags in flags:
            for flag, severity, note in author_flags:
                icon = "🔴" if severity == "high" else "🟡" if severity == "medium" else "🟢"
                st.markdown(
                    f'{icon} **{author}** — {flag}'
                    f'<span style="color:#666;font-size:0.82em;margin-left:8px">({note})</span>',
                    unsafe_allow_html=True,
                )

    # Consistency score per author
    st.markdown("---")
    st.markdown("**Writing Consistency Score:**")
    for p in profiles:
        # Simple consistency: based on how regular the metrics are
        ttr_var = abs(p["avg_ttr"] - p["vocab_richness"])
        consistency = max(0, 100 - ttr_var * 100 - abs(p["hapax_ratio"] - 0.4) * 50)
        consistency = min(max(consistency, 20), 98)
        color = "#28a745" if consistency >= 70 else "#ffc107" if consistency >= 50 else "#dc3545"
        st.markdown(
            f'<div style="display:flex;align-items:center;margin:3px 0">'
            f'<span style="width:130px;font-size:0.88em;font-weight:600">{p["author"]}</span>'
            f'<div style="width:40%;background:#e0e0e0;border-radius:4px;height:14px">'
            f'<div style="width:{consistency}%;background:{color};border-radius:4px;height:100%"></div></div>'
            f'<span style="margin-left:8px;font-size:0.85em;color:{color};font-weight:600">{consistency:.0f}/100</span></div>',
            unsafe_allow_html=True,
        )


def _render_document_analysis(docs: list[dict]):
    """Render per-document authorship scoring."""
    st.subheader("📄 Document Authorship Scores")

    rows = []
    for d in docs:
        rows.append({
            "ID": d["id"],
            "Title": d["title"],
            "Author": d["author"],
            "Department": d["department"],
            "Words": d["word_count"],
            "Author Match": f"{d['similarity_to_author']}%",
            "Flagged": "🚨" if d["flagged"] else "✅",
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Flagged documents
    flagged = [d for d in docs if d["flagged"]]
    if flagged:
        st.warning(f"⚠️ **{len(flagged)} documents** have authorship match below 75% — possible ghost-writing or style mismatch.")
        for d in flagged:
            st.markdown(
                f'- 🚨 **{d["title"]}** by {d["author"]} — {d["similarity_to_author"]}% match',
            )


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------

def render_writing_style_analyzer():
    """Render the Writing Style Analyzer page."""
    st.title("✍️ Writing Style Analyzer")
    st.markdown(
        "Analyze writing style fingerprints, compare authorship patterns, and detect style anomalies."
    )

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.subheader("Analysis Mode")
        mode = st.radio(
            "Mode",
            ["Demo (Mock Data)", "Custom Text Input"],
            index=0,
        )

        if mode == "Custom Text Input":
            st.markdown("---")
            st.subheader("📝 Input Documents")
            num_docs = st.number_input("Number of documents to compare", 2, 5, 2)
            custom_docs = []
            for i in range(num_docs):
                author = st.text_input(f"Author {i+1}", value=f"Author {i+1}", key=f"auth_{i}")
                text = st.text_area(
                    f"Text for {author}",
                    value=f"Sample text for {author}. This is a placeholder to demonstrate the style analysis system. The actual analysis would process the full uploaded document.",
                    height=120,
                    key=f"text_{i}",
                )
                custom_docs.append({"author": author, "text": text})

        st.markdown("---")
        st.subheader("📊 Display Options")
        show_profiles = st.checkbox("Show Author Profiles", True)
        show_vocab = st.checkbox("Vocabulary Analysis", True)
        show_sentence = st.checkbox("Sentence Structure", True)
        show_punctuation = st.checkbox("Punctuation Patterns", True)
        show_pronouns = st.checkbox("Pronoun Analysis", True)
        show_comparison = st.checkbox("Authorship Comparison", True)
        show_forensic = st.checkbox("Forensic Flags", True)
        show_docs = st.checkbox("Document Analysis", True)

    # Load data
    if mode == "Demo (Mock Data)":
        profiles = _generate_mock_style_profiles()
        docs = _generate_mock_documents()
    else:
        profiles = []
        for cd in custom_docs:
            profile = _compute_style_profile(cd["text"], cd["author"])
            profiles.append(profile)
        docs = []

    if not profiles:
        st.warning("No data to analyze.")
        return

    # Render sections
    if show_profiles:
        _render_overview(profiles)

    if show_vocab:
        st.markdown("---")
        _render_vocabulary_analysis(profiles)

    if show_sentence:
        st.markdown("---")
        _render_sentence_analysis(profiles)

    if show_punctuation:
        st.markdown("---")
        _render_punctuation_analysis(profiles)

    if show_pronouns:
        st.markdown("---")
        _render_pronoun_analysis(profiles)

    if show_comparison and len(profiles) >= 2:
        st.markdown("---")
        _render_authorship_comparison(profiles)

    if show_forensic:
        st.markdown("---")
        _render_forensic_flags(profiles)

    if show_docs and docs:
        st.markdown("---")
        _render_document_analysis(docs)

    # Footer
    st.markdown("---")
    st.caption(
        f"Writing Style Analyzer | {len(profiles)} author profiles analyzed | "
        f"{len(docs)} documents | Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )


# Entry point
if __name__ == "__main__" or True:
    render_writing_style_analyzer()
