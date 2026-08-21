"""Streamlit Component for rendering FAISS Vector Search Cards."""

from typing import Any, Dict


def render_vector_match_card(match: Dict[str, Any]) -> str:
    """Renders HTML glassmorphic markup for displaying vector match details."""
    sim_pct = int(match.get("cosine_similarity_score", 0.0) * 100)
    l2_dist = match.get("l2_distance", 0.0)
    rank = match.get("rank_position", 1)

    badge_color = (
        "#10B981" if sim_pct > 80 else "#F59E0B" if sim_pct > 50 else "#6366F1"
    )

    return f"""
    <div style="
        background: rgba(15, 23, 42, 0.9);
        border: 1px solid rgba(51, 65, 85, 1);
        border-radius: 20px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
    ">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
            <span style="
                background: rgba(99, 102, 241, 0.15);
                border: 1px solid rgba(99, 102, 241, 0.4);
                color: #A5B4FC;
                font-size: 11px;
                font-weight: 800;
                padding: 4px 12px;
                border-radius: 9999px;
            ">
                Rank #{rank} k-NN Result
            </span>
            <span style="
                background: {badge_color}20;
                border: 1px solid {badge_color}50;
                color: {badge_color};
                font-size: 11px;
                font-weight: 800;
                padding: 4px 12px;
                border-radius: 9999px;
            ">
                {sim_pct}% Cosine Similarity
            </span>
        </div>
        
        <h4 style="color: white; font-weight: 900; margin: 0 0 8px 0;">
            Document: {match.get("matched_document_title")} ({match.get("matched_chunk_id")})
        </h4>
        
        <p style="color: #94A3B8; font-size: 13px; font-style: italic; margin-bottom: 16px; background: rgba(2, 6, 23, 0.5); padding: 12px; border-radius: 12px; border: 1px solid rgba(30, 41, 59, 0.8);">
            "{match.get("matched_text_snippet")}"
        </p>

        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
            <div style="background: rgba(2, 6, 23, 0.6); padding: 12px; border-radius: 12px; border: 1px solid rgba(30, 41, 59, 1);">
                <span style="color: #94A3B8; font-size: 11px;">Inner Product Score:</span>
                <div style="color: #10B981; font-weight: 800; font-size: 16px;">{match.get("cosine_similarity_score")}</div>
            </div>
            <div style="background: rgba(2, 6, 23, 0.6); padding: 12px; border-radius: 12px; border: 1px solid rgba(30, 41, 59, 1);">
                <span style="color: #94A3B8; font-size: 11px;">L2 Distance:</span>
                <div style="color: #6366F1; font-weight: 800; font-size: 16px;">{l2_dist}</div>
            </div>
        </div>
    </div>
    """
