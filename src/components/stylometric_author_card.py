"""Streamlit Component for rendering Stylometric Author Attribution Cards."""

from typing import Any, Dict


def render_stylometric_card(match: Dict[str, Any]) -> str:
    """Renders HTML glassmorphic markup for displaying author attribution metrics."""
    confidence = match.get("attribution_confidence_percentage", 0.0)
    distance = match.get("stylometric_distance", 0.0)
    is_same = match.get("is_same_author", False)

    badge_color = "#10B981" if is_same else "#6366F1"
    status_text = "SAME AUTHOR VERIFIED" if is_same else "DIFFERENT AUTHOR / STYLE"

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
                background: rgba(16, 185, 129, 0.15);
                border: 1px solid rgba(16, 185, 129, 0.4);
                color: #6EE7B7;
                font-size: 11px;
                font-weight: 800;
                padding: 4px 12px;
                border-radius: 9999px;
            ">
                Candidate: {match.get("candidate_author_alias")}
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
                {status_text} ({confidence}% Conf)
            </span>
        </div>
        
        <h4 style="color: white; font-weight: 900; margin: 0 0 8px 0;">
            Query Doc: {match.get("query_document_id")} ↔ Candidate Doc: {match.get("candidate_document_id")}
        </h4>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 16px;">
            <div style="background: rgba(2, 6, 23, 0.6); padding: 12px; border-radius: 12px; border: 1px solid rgba(30, 41, 59, 1);">
                <span style="color: #94A3B8; font-size: 11px;">Stylometric Distance:</span>
                <div style="color: #6366F1; font-weight: 800; font-size: 16px;">{distance}</div>
            </div>
            <div style="background: rgba(2, 6, 23, 0.6); padding: 12px; border-radius: 12px; border: 1px solid rgba(30, 41, 59, 1);">
                <span style="color: #94A3B8; font-size: 11px;">Dominant Trait:</span>
                <div style="color: #10B981; font-weight: 800; font-size: 14px;">{match.get("dominant_stylometric_trait")}</div>
            </div>
        </div>
    </div>
    """
