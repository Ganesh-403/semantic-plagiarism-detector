"""
app/pages/7_AI_Content_Detection.py
------------------------------------
Streamlit multi-page app: AI-Generated Content Detection.

Detects AI-generated text with confidence scoring, model fingerprinting,
statistical analysis, and per-document AI probability assessment.
"""

import math
import re
from collections import Counter
from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="AI Content Detection - Plagiarism Detector",
    page_icon="🤖",
    layout="wide",
)

# ---------------------------------------------------------------------------
# AI Detection Engine (statistical heuristics)
# ---------------------------------------------------------------------------

def _compute_ai_indicators(text: str) -> dict[str, Any]:
    """Compute statistical indicators for AI-generated content detection."""
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    words = text.split()
    total_words = len(words)
    unique_words = set(w.lower() for w in words if w.isalpha())

    # 1. Perplexity proxy: burstiness (variance of sentence lengths)
    sent_lengths = [len(s.split()) for s in sentences]
    avg_sl = sum(sent_lengths) / len(sent_lengths) if sent_lengths else 0
    variance = sum((sl - avg_sl) ** 2 for sl in sent_lengths) / len(sent_lengths) if sent_lengths else 0
    burstiness = math.sqrt(variance) if variance > 0 else 0
    # Human text has higher burstiness; AI text is more uniform
    burstiness_score = max(0, min(100, 100 - (burstiness / avg_sl * 100) if avg_sl else 50))

    # 2. Vocabulary diversity (TTR)
    vocab_richness = len(unique_words) / total_words if total_words else 0

    # 3. Repetition patterns
    bigrams = [f"{words[i].lower()} {words[i+1].lower()}" for i in range(len(words) - 1)]
    bigram_freq = Counter(bigrams)
    repeated_bigrams = sum(1 for v in bigram_freq.values() if v > 2)
    repetition_score = min(100, repeated_bigrams / max(len(bigrams), 1) * 500)

    # 4. Sentence structure uniformity
    sent_starts = []
    for s in sentences:
        parts = s.split()
        if parts:
            sent_starts.append(parts[0].lower())
    starter_freq = Counter(sent_starts)
    top_starter_pct = (starter_freq.most_common(1)[0][1] / len(sent_starts) * 100) if sent_starts else 0
    uniformity_score = min(100, top_starter_pct * 2)

    # 5. Filler word density (AI tends to use more hedging)
    fillers = [
        "furthermore", "moreover", "additionally", "consequently", "nevertheless",
        "however", "indeed", "notably", "significantly", "subsequently",
        "it is worth noting", "in conclusion", "as a result", "in addition",
        "overall", "essentially", "fundamentally", "importantly", "remarkably",
    ]
    filler_count = sum(1 for f in fillers if f in text.lower())
    filler_density = filler_count / max(len(sentences), 1) * 10

    # 6. Punctuation regularity (AI uses very consistent punctuation)
    comma_rate = text.count(',') / max(len(sentences), 1)
    period_rate = 1.0  # every sentence ends with period
    semicolon_rate = text.count(';') / max(len(sentences), 1)
    # Human text varies more
    punct_regularity = abs(comma_rate - 1.5) < 0.3 and semicolon_rate < 0.1

    # 7. Word length distribution (AI tends toward medium-length words)
    word_lens = [len(w) for w in words if w.isalpha()]
    if word_lens:
        avg_wl = sum(word_lens) / len(word_lens)
        wl_std = math.sqrt(sum((w - avg_wl) ** 2 for w in word_lens) / len(word_lens))
    else:
        avg_wl, wl_std = 5, 1
    # Lower std = more uniform = more AI-like
    wl_uniformity = max(0, min(100, 100 - wl_std * 20))

    # 8. Paragraph structure (AI tends to have uniform paragraph lengths)
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    if paragraphs:
        para_lengths = [len(p.split()) for p in paragraphs]
        avg_para = sum(para_lengths) / len(para_lengths)
        para_variance = sum((pl - avg_para) ** 2 for pl in para_lengths) / len(para_lengths) if para_lengths else 0
        para_burstiness = math.sqrt(para_variance) if para_variance > 0 else 0
        para_uniformity = max(0, min(100, 100 - (para_burstiness / avg_para * 100) if avg_para else 50))
    else:
        para_uniformity = 50

    # 9. Lexical overlap score (AI reuses common phrases)
    phrases_3gram = [f"{words[i].lower()} {words[i+1].lower()} {words[i+2].lower()}"
                     for i in range(len(words) - 2)]
    trigram_freq = Counter(phrases_3gram)
    repeated_3grams = sum(1 for v in trigram_freq.values() if v > 1)
    overlap_score = min(100, repeated_3grams / max(len(phrases_3gram), 1) * 300)

    # Combined AI probability (weighted heuristic)
    ai_probability = (
        burstiness_score * 0.20 +
        repetition_score * 0.10 +
        uniformity_score * 0.12 +
        filler_density * 10 * 0.13 +
        (80 if punct_regularity else 30) * 0.10 +
        wl_uniformity * 0.10 +
        para_uniformity * 0.10 +
        overlap_score * 0.10 +
        (100 - vocab_richness * 100) * 0.05
    )
    ai_probability = max(0, min(100, ai_probability))

    return {
        "ai_probability": round(ai_probability, 1),
        "burstiness_score": round(burstiness_score, 1),
        "repetition_score": round(repetition_score, 1),
        "uniformity_score": round(uniformity_score, 1),
        "filler_density": round(filler_density, 1),
        "punct_regularity": punct_regularity,
        "wl_uniformity": round(wl_uniformity, 1),
        "para_uniformity": round(para_uniformity, 1),
        "overlap_score": round(overlap_score, 1),
        "vocab_richness": round(vocab_richness, 3),
        "sentence_count": len(sentences),
        "word_count": total_words,
        "unique_words": len(unique_words),
        "avg_sentence_length": round(avg_sl, 1),
        "burstiness_raw": round(burstiness, 2),
        "filler_words_found": filler_count,
    }


def _classify_ai_probability(prob: float) -> tuple[str, str]:
    """Classify AI probability into a label and color."""
    if prob >= 80:
        return "Very Likely AI", "#dc3545"
    elif prob >= 60:
        return "Likely AI", "#fd7e14"
    elif prob >= 40:
        return "Uncertain", "#ffc107"
    elif prob >= 20:
        return "Likely Human", "#28a745"
    else:
        return "Very Likely Human", "#20c997"


# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------

def _generate_mock_detections() -> list[dict[str, Any]]:
    """Generate mock AI detection results."""
    import random
    random.seed(55)

    doc_names = [
        "Thesis_Chapter3.docx", "Research_Paper_AI.docx", "Literature_Review.pdf",
        "Capstone_Project.docx", "Seminar_Report.pdf", "Lab_Report_Physics.docx",
        "Dissertation_Draft.docx", "Conference_Paper.docx", "Assignment_Week5.docx",
        "Midterm_Essay.docx", "Group_Project_Report.pdf", "Technical_Writeup.docx",
        "Case_Study_Analysis.docx", "Term_Paper_History.docx", "Methodology_Section.docx",
        "Abstract_Collection.pdf", "Appendix_Draft.docx", "Review_Article.docx",
        "Survey_Results.docx", "Final_Proposal.docx", "Preprint_v2.pdf",
        "Book_Chapter_Draft.docx", "Workshop_Paper.docx", "Thesis_Intro.docx",
        "Dataset_Analysis.docx", "Proposal_v3.docx", "Research_Notes.pdf",
        "Presentation_Slides.docx", "Lab_Notes.pdf", "Essay_Draft.docx",
    ]
    authors = [
        "Alice Johnson", "Bob Smith", "Carol White", "David Brown", "Eva Martinez",
        "Frank Lee", "Grace Kim", "Henry Wilson", "Iris Chen", "Jack Davis",
    ]
    ai_models = [
        "GPT-4", "GPT-3.5", "Claude 3", "Gemini Pro", "Llama 2",
        "Mistral 7B", "PaLM 2", "Cohere", None, None, None,
    ]
    departments = [
        "Computer Science", "Electrical Engineering", "Physics", "Mathematics",
        "Biology", "Chemistry", "Data Science", "Mechanical Engineering",
    ]

    detections = []
    for i in range(30):
        prob = round(random.uniform(5, 98), 1)
        label, color = _classify_ai_probability(prob)
        model = random.choice(ai_models) if prob > 50 else None
        detections.append({
            "id": f"AI-{5000 + i}",
            "document": doc_names[i],
            "author": random.choice(authors),
            "department": random.choice(departments),
            "word_count": random.randint(500, 8000),
            "ai_probability": prob,
            "label": label,
            "color": color,
            "detected_model": model,
            "confidence": round(random.uniform(60, 99), 1),
            "burstiness_score": round(random.uniform(10, 85), 1),
            "repetition_score": round(random.uniform(5, 70), 1),
            "uniformity_score": round(random.uniform(15, 90), 1),
            "filler_density": round(random.uniform(0, 8), 1),
            "overlap_score": round(random.uniform(10, 80), 1),
            "vocab_richness": round(random.uniform(0.3, 0.75), 3),
            "analyst_notes": random.choice([
                "High AI probability — multiple indicators present",
                "Moderate confidence — needs manual review",
                "Low AI indicators — likely human-written",
                "Strong statistical evidence of AI generation",
                "Pattern consistent with GPT-4 output",
                "",
            ]),
            "scanned_at": (datetime.now() - random.randint(0, 30) * __import__('datetime').timedelta(days=1)).strftime("%Y-%m-%d %H:%M"),
        })
    return detections


# ---------------------------------------------------------------------------
# Chart renderers
# ---------------------------------------------------------------------------

def _render_confidence_gauge(probability: float, size: int = 120):
    """Render a circular confidence gauge."""
    angle = (probability / 100) * 360
    rad = math.pi * angle / 180
    r = size * 0.38
    cx, cy = size / 2, size / 2
    x_end = cx + r * math.sin(rad)
    y_end = cy - r * math.cos(rad)
    large_arc = 1 if angle > 180 else 0
    label, color = _classify_ai_probability(probability)

    svg = f'''<svg width="{size}" height="{size + 20}" viewBox="0 0 {size} {size + 20}" xmlns="http://www.w3.org/2000/svg">
  <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="#2a2a3e" stroke-width="10"/>
  <path d="M {cx} {cy - r} A {r} {r} 0 {large_arc} 1 {x_end:.1f} {y_end:.1f}"
        fill="none" stroke="{color}" stroke-width="10" stroke-linecap="round"/>
  <text x="{cx}" y="{cy + 3}" text-anchor="middle" font-size="{size // 5}" font-weight="bold" fill="{color}">{probability:.0f}%</text>
  <text x="{cx}" y="{cy + size * 0.22}" text-anchor="middle" font-size="{size // 10}" fill="#aaa">{label}</text>
</svg>'''
    return svg


def _render_indicator_bar(name: str, value: float, max_val: float = 100, color: str = "#4a90d9"):
    """Render a single indicator bar."""
    pct = min(value / max_val * 100, 100)
    return (
        f'<div style="display:flex;align-items:center;margin:4px 0">'
        f'<span style="width:130px;font-size:0.85em">{name}</span>'
        f'<div style="width:55%;background:#1e1e2e;border-radius:4px;height:16px">'
        f'<div style="width:{pct:.0f}%;background:{color};border-radius:4px;height:100%"></div></div>'
        f'<span style="margin-left:8px;font-size:0.85em;font-weight:600;color:{color}">{value}</span></div>'
    )


def _render_model_fingerprint_bar(detections: list[dict]):
    """Render model detection distribution."""
    model_counts = {}
    for d in detections:
        model = d.get("detected_model") or "None"
        model_counts[model] = model_counts.get(model, 0) + 1

    st.markdown("**Detected AI Model Distribution:**")
    colors = ["#4a90d9", "#e83e8c", "#28a745", "#fd7e14", "#6f42c1", "#20c997", "#dc3545", "#ffc107", "#6c757d"]
    max_count = max(model_counts.values()) if model_counts else 1
    for idx, (model, count) in enumerate(sorted(model_counts.items(), key=lambda x: x[1], reverse=True)):
        pct = count / max_count * 100
        color = colors[idx % len(colors)]
        st.markdown(
            _render_indicator_bar(model, count, max_count, color),
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------

def _render_detection_overview(detections: list[dict]):
    """Render overview KPIs and summary."""
    st.subheader("🤖 AI Detection Overview")

    total = len(detections)
    likely_ai = sum(1 for d in detections if d["ai_probability"] >= 60)
    uncertain = sum(1 for d in detections if 40 <= d["ai_probability"] < 60)
    likely_human = sum(1 for d in detections if d["ai_probability"] < 40)
    avg_prob = sum(d["ai_probability"] for d in detections) / total if total else 0
    high_conf = sum(1 for d in detections if d["confidence"] > 90 and d["ai_probability"] > 60)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Documents Scanned", total)
    c2.metric("🤖 Likely AI", likely_ai, delta=None, delta_color="inverse")
    c3.metric("❓ Uncertain", uncertain)
    c4.metric("👤 Likely Human", likely_human)
    c5.metric("Avg AI Score", f"{avg_prob:.0f}%")
    c6.metric("High Confidence AI", high_conf)

    st.markdown(
        f"Out of **{total} documents** scanned, **{likely_ai}** are classified as likely AI-generated, "
        f"**{uncertain}** are uncertain (need manual review), and **{likely_human}** are likely human-written. "
        f"The average AI probability is **{avg_prob:.1f}%**."
    )

    if likely_ai > 0:
        st.warning(f"⚠️ **{likely_ai} documents** show strong AI-generation signals. {high_conf} have high-confidence model matches.")


def _render_probability_distribution(detections: list[dict]):
    """Render AI probability distribution chart."""
    st.subheader("📊 Probability Distribution")

    # Histogram bins
    bins = {"0-10%": 0, "10-20%": 0, "20-30%": 0, "30-40%": 0, "40-50%": 0,
            "50-60%": 0, "60-70%": 0, "70-80%": 0, "80-90%": 0, "90-100%": 0}
    for d in detections:
        p = d["ai_probability"]
        idx = min(int(p // 10), 9)
        key = list(bins.keys())[idx]
        bins[key] += 1

    max_val = max(bins.values()) if bins else 1
    colors_list = ["#20c997", "#28a745", "#28a745", "#ffc107", "#ffc107",
                   "#fd7e14", "#fd7e14", "#dc3545", "#dc3545", "#dc3545"]

    chart_html = '<div style="display:flex;align-items:flex-end;gap:6px;height:200px;padding:10px 0">'
    for i, (bin_label, count) in enumerate(bins.items()):
        h = (count / max_val * 180) if max_val else 0
        chart_html += (
            f'<div style="flex:1;display:flex;flex-direction:column;align-items:center">'
            f'<span style="font-size:0.75em;color:#ccc;margin-bottom:4px">{count}</span>'
            f'<div style="width:100%;height:{h:.0f}px;background:{colors_list[i]};border-radius:4px 4px 0 0;min-height:2px"></div>'
            f'<span style="font-size:0.7em;color:#888;margin-top:4px;writing-mode:vertical-lr;transform:rotate(180deg);height:60px">{bin_label}</span>'
            f'</div>'
        )
    chart_html += '</div>'
    st.markdown(chart_html, unsafe_allow_html=True)

    # Color legend
    st.markdown(
        '<div style="font-size:0.82em;color:#888;margin-top:8px">'
        '<span style="color:#20c997">■</span> Human (0-30%) &nbsp;&nbsp; '
        '<span style="color:#ffc107">■</span> Uncertain (30-60%) &nbsp;&nbsp; '
        '<span style="color:#dc3545">■</span> AI (60-100%)'
        '</div>',
        unsafe_allow_html=True,
    )


def _render_indicator_analysis(detections: list[dict]):
    """Render detailed indicator analysis."""
    st.subheader("🔬 Statistical Indicators")

    # Average indicators across all documents
    metrics = [
        ("burstiness_score", "Burstiness (sentence length variance)", "Human text is more bursty"),
        ("repetition_score", "Repetition Patterns", "AI repeats phrases more"),
        ("uniformity_score", "Sentence Start Uniformity", "AI starts sentences similarly"),
        ("filler_density", "Filler Word Density", "AI uses more hedging words"),
        ("wl_uniformity", "Word Length Uniformity", "AI uses more uniform word lengths"),
        ("para_uniformity", "Paragraph Uniformity", "AI has uniform paragraph sizes"),
        ("overlap_score", "Lexical Overlap", "AI reuses common phrases"),
    ]

    avg_vals = {}
    for key, label, desc in metrics:
        avg_vals[key] = sum(d.get(key, 0) for d in detections) / len(detections) if detections else 0

    st.markdown("**Average Indicator Scores (all documents):**")
    for key, label, desc in metrics:
        val = avg_vals[key]
        color = "#dc3545" if val > 70 else "#fd7e14" if val > 50 else "#ffc107" if val > 30 else "#28a745"
        st.markdown(_render_indicator_bar(label, val, 100, color), unsafe_allow_html=True)
        st.caption(desc)


def _render_document_table(detections: list[dict], min_prob: float = 0, max_prob: float = 100):
    """Render document detection table with filters."""
    st.subheader("📄 Document Detection Results")

    filtered = [d for d in detections if min_prob <= d["ai_probability"] <= max_prob]
    if not filtered:
        st.info("No documents match the current filters.")
        return

    rows = []
    for d in sorted(filtered, key=lambda x: x["ai_probability"], reverse=True):
        rows.append({
            "ID": d["id"],
            "Document": d["document"][:25],
            "Author": d["author"],
            "AI %": d["ai_probability"],
            "Confidence": f"{d['confidence']}%",
            "Label": d["label"],
            "Model": d["detected_model"] or "—",
            "Burstiness": d["burstiness_score"],
            "Repetition": d["repetition_score"],
            "Status": "🚨" if d["ai_probability"] >= 70 else "⚠️" if d["ai_probability"] >= 50 else "✅",
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Export
    csv = pd.DataFrame(rows).to_csv(index=False)
    st.download_button(
        "📥 Export Results (CSV)", csv,
        file_name=f"ai_detection_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )


def _render_document_deep_dive(detections: list[dict]):
    """Render detailed analysis for individual documents."""
    st.subheader("🔍 Document Deep Dive")

    doc_options = [f"{d['id']} — {d['document'][:30]}" for d in sorted(detections, key=lambda x: x["ai_probability"], reverse=True)]
    selected = st.selectbox("Select Document", ["None"] + doc_options)

    if selected == "None":
        st.info("Select a document above to see detailed analysis.")
        return

    doc_id = selected.split(" — ")[0]
    doc = next((d for d in detections if d["id"] == doc_id), None)
    if not doc:
        return

    # Header with gauge
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown(_render_confidence_gauge(doc["ai_probability"]), unsafe_allow_html=True)
    with c2:
        st.markdown(f"### {doc['document']}")
        st.markdown(
            f"**Author:** {doc['author']} | **Dept:** {doc['department']} | "
            f"**Words:** {doc['word_count']:,} | **Scanned:** {doc['scanned_at']}"
        )
        if doc["detected_model"]:
            st.markdown(f"**Detected Model:** 🤖 {doc['detected_model']} (confidence: {doc['confidence']}%)")
        if doc["analyst_notes"]:
            st.info(f"📝 {doc['analyst_notes']}")

    st.markdown("---")

    # Indicator breakdown
    indicators = [
        ("Burstiness", doc["burstiness_score"], "High = more human-like"),
        ("Repetition", doc["repetition_score"], "High = more AI-like"),
        ("Sentence Uniformity", doc["uniformity_score"], "High = more AI-like"),
        ("Filler Density", doc["filler_density"], "High = more AI-like"),
        ("Word Length Uniformity", doc["wl_uniformity"], "High = more AI-like"),
        ("Paragraph Uniformity", doc["para_uniformity"], "High = more AI-like"),
        ("Lexical Overlap", doc["overlap_score"], "High = more AI-like"),
        ("Vocab Richness", doc["vocab_richness"] * 100, "High = more human-like"),
    ]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**AI Indicators (higher = more AI-like):**")
        for name, value, hint in indicators[:4]:
            color = "#dc3545" if value > 70 else "#fd7e14" if value > 50 else "#28a745"
            st.markdown(_render_indicator_bar(name, value, 100, color), unsafe_allow_html=True)
            st.caption(hint)

    with c2:
        st.markdown("**Additional Metrics:**")
        for name, value, hint in indicators[4:]:
            color = "#dc3545" if value > 70 else "#fd7e14" if value > 50 else "#28a745"
            st.markdown(_render_indicator_bar(name, value, 100, color), unsafe_allow_html=True)
            st.caption(hint)

    # Classification
    label, color = _classify_ai_probability(doc["ai_probability"])
    st.markdown(
        f'<div style="border:2px solid {color};border-radius:8px;padding:12px;margin:12px 0;background:#1e1e2e;text-align:center">'
        f'<span style="font-size:1.4em;color:{color};font-weight:700">{label}</span><br/>'
        f'<span style="color:#aaa">AI Probability: {doc["ai_probability"]}% | Confidence: {doc["confidence"]}%</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _render_model_analysis(detections: list[dict]):
    """Render AI model detection analysis."""
    st.subheader("🧠 Model Fingerprinting")

    with_model = [d for d in detections if d.get("detected_model")]
    st.markdown(f"**{len(with_model)} documents** have a detected AI model signature.")

    _render_model_fingerprint_bar(detections)

    # Model-specific stats
    model_groups: dict[str, list[dict]] = {}
    for d in with_model:
        model_groups.setdefault(d["detected_model"], []).append(d)

    st.markdown("**Model Breakdown:**")
    for model, docs in sorted(model_groups.items(), key=lambda x: len(x[1]), reverse=True):
        avg_prob = sum(d["ai_probability"] for d in docs) / len(docs)
        avg_conf = sum(d["confidence"] for d in docs) / len(docs)
        st.markdown(
            f'<div style="border-left:4px solid #4a90d9;padding:8px 12px;margin:6px 0;background:#1e1e2e;border-radius:4px">'
            f'<strong>🤖 {model}</strong> — {len(docs)} documents<br/>'
            f'<span style="font-size:0.85em;color:#aaa">Avg AI Probability: {avg_prob:.0f}% | Avg Confidence: {avg_conf:.0f}%</span><br/>'
            f'<span style="font-size:0.82em;color:#888">Authors: {", ".join(set(d["author"] for d in docs))}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


def _render_trend_over_time(detections: list[dict]):
    """Render AI detection trends over time."""
    st.subheader("📈 Detection Trends")

    # Group by date
    daily = {}
    for d in detections:
        day = d["scanned_at"][:10]
        daily.setdefault(day, {"total": 0, "ai": 0, "human": 0})
        daily[day]["total"] += 1
        if d["ai_probability"] >= 60:
            daily[day]["ai"] += 1
        else:
            daily[day]["human"] += 1

    if not daily:
        st.info("No trend data.")
        return

    sorted_days = sorted(daily.keys())
    labels = [d[5:] for d in sorted_days]
    ai_vals = [daily[d]["ai"] for d in sorted_days]
    human_vals = [daily[d]["human"] for d in sorted_days]
    total_vals = [daily[d]["total"] for d in sorted_days]

    max_val = max(total_vals) if total_vals else 1

    st.markdown("**Daily Scan Results:**")
    chart_html = '<div style="display:flex;align-items:flex-end;gap:4px;height:160px;padding:8px 0">'
    for i, day in enumerate(sorted_days):
        t = total_vals[i]
        ai = ai_vals[i]
        human = human_vals[i]
        h_total = (t / max_val * 140) if max_val else 0
        h_ai = (ai / max_val * 140) if max_val else 0
        chart_html += (
            f'<div style="flex:1;display:flex;flex-direction:column;align-items:center">'
            f'<div style="width:100%;display:flex;flex-direction:column-reverse">'
            f'<div style="width:100%;height:{h_ai:.0f}px;background:#dc3545;border-radius:0 0 3px 3px"></div>'
            f'<div style="width:100%;height:{(h_total - h_ai):.0f}px;background:#28a745;border-radius:3px 3px 0 0"></div>'
            f'</div>'
            f'<span style="font-size:0.65em;color:#888;writing-mode:vertical-lr;transform:rotate(180deg);margin-top:4px;height:40px">{labels[i]}</span>'
            f'</div>'
        )
    chart_html += '</div>'
    st.markdown(chart_html, unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:0.82em;color:#888">'
        '<span style="color:#28a745">■</span> Human &nbsp;&nbsp; '
        '<span style="color:#dc3545">■</span> AI-Generated'
        '</div>',
        unsafe_allow_html=True,
    )


def _render_recommendations(detections: list[dict]):
    """Render AI detection recommendations."""
    st.subheader("💡 Recommendations & Actions")

    very_likely = [d for d in detections if d["ai_probability"] >= 80]
    likely = [d for d in detections if 60 <= d["ai_probability"] < 80]
    uncertain = [d for d in detections if 40 <= d["ai_probability"] < 60]

    if very_likely:
        st.error(
            f"🚨 **{len(very_likely)} documents** are very likely AI-generated (≥80% probability). "
            f"Immediate academic integrity review recommended."
        )
        for d in very_likely[:5]:
            st.markdown(f"- **{d['document']}** by {d['author']} — {d['ai_probability']}% AI | Model: {d.get('detected_model', 'Unknown')}")

    if likely:
        st.warning(
            f"⚠️ **{len(likely)} documents** are likely AI-generated (60-80%). "
            f"Secondary verification recommended."
        )
        for d in likely[:5]:
            st.markdown(f"- **{d['document']}** by {d['author']} — {d['ai_probability']}% AI")

    if uncertain:
        st.info(
            f"❓ **{len(uncertain)} documents** are uncertain (40-60%). "
            f"Manual review recommended to determine authorship."
        )

    # Action items
    st.markdown("**Recommended Actions:**")
    actions = [
        {"priority": "🔴 Immediate", "action": f"Review {len(very_likely)} high-probability AI documents", "deadline": "24 hours"},
        {"priority": "🟠 High", "action": f"Verify {len(likely)} likely AI documents with authors", "deadline": "48 hours"},
        {"priority": "🟡 Medium", "action": f"Manually review {len(uncertain)} uncertain documents", "deadline": "1 week"},
        {"priority": "🟢 Preventive", "action": "Implement AI usage disclosure policy", "deadline": "Next semester"},
        {"priority": "🟢 Preventive", "action": "Deploy real-time AI detection in submission pipeline", "deadline": "2 weeks"},
    ]

    for act in actions:
        st.markdown(
            f'<div style="border:1px solid #333;border-radius:6px;padding:8px 12px;margin:6px 0;background:#1e1e2e">'
            f'<strong>{act["priority"]}</strong> — {act["action"]}<br/>'
            f'<span style="font-size:0.82em;color:#888">Deadline: {act["deadline"]}</span></div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------

def render_ai_content_detection():
    """Render the AI Content Detection page."""
    st.title("🤖 AI Content Detection")
    st.markdown(
        "Detect AI-generated text with statistical analysis, confidence scoring, and model fingerprinting."
    )

    detections = _generate_mock_detections()

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Detection Settings")
        min_prob = st.slider("Min AI Probability", 0, 100, 0)
        max_prob = st.slider("Max AI Probability", 0, 100, 100)
        min_confidence = st.slider("Min Confidence", 0, 100, 0)

        st.markdown("---")
        st.subheader("📊 Sections")
        show_overview = st.checkbox("Detection Overview", True)
        show_distribution = st.checkbox("Probability Distribution", True)
        show_indicators = st.checkbox("Statistical Indicators", True)
        show_table = st.checkbox("Document Results", True)
        show_deep = st.checkbox("Document Deep Dive", True)
        show_model = st.checkbox("Model Fingerprinting", True)
        show_trend = st.checkbox("Detection Trends", True)
        show_recommendations = st.checkbox("Recommendations", True)

    # Apply filters
    filtered = [d for d in detections if min_prob <= d["ai_probability"] <= max_prob and d["confidence"] >= min_confidence]

    if show_overview:
        _render_detection_overview(filtered)

    if show_distribution:
        st.markdown("---")
        _render_probability_distribution(filtered)

    if show_indicators:
        st.markdown("---")
        _render_indicator_analysis(filtered)

    if show_table:
        st.markdown("---")
        _render_document_table(detections, min_prob, max_prob)

    if show_deep:
        st.markdown("---")
        _render_document_deep_dive(detections)

    if show_model:
        st.markdown("---")
        _render_model_analysis(filtered)

    if show_trend:
        st.markdown("---")
        _render_trend_over_time(detections)

    if show_recommendations:
        st.markdown("---")
        _render_recommendations(filtered)

    st.markdown("---")
    st.caption(
        f"AI Content Detection | {len(filtered)} documents analyzed | "
        f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )


# Entry point
if __name__ == "__main__" or True:
    render_ai_content_detection()
