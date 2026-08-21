"""Streamlit Component for rendering Multimodal OCR Plagiarism Cards."""

from typing import Any, Dict


def render_ocr_match_card(match: Dict[str, Any]) -> str:
    """Renders HTML glassmorphic markup for displaying multimodal OCR scan metrics."""
    overall_pct = int(match.get("overall_multimodal_score", 0.0) * 100)
    ocr_pct = int(match.get("ocr_text_similarity", 0.0) * 100)
    layout_pct = int(match.get("layout_structure_similarity", 0.0) * 100)

    badge_color = (
        "#EF4444" if overall_pct > 75 else "#F59E0B" if overall_pct > 40 else "#10B981"
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
                background: rgba(16, 185, 129, 0.15);
                border: 1px solid rgba(16, 185, 129, 0.4);
                color: #6EE7B7;
                font-size: 11px;
                font-weight: 800;
                padding: 4px 12px;
                border-radius: 9999px;
            ">
                OCR Engine: {match.get("ocr_engine_used")}
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
                {overall_pct}% Multimodal Score
            </span>
        </div>
        
        <h4 style="color: white; font-weight: 900; margin: 0 0 8px 0;">
            Image: {match.get("source_image_name")} ↔ Target: {match.get("target_reference_title")}
        </h4>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 16px;">
            <div style="background: rgba(2, 6, 23, 0.6); padding: 12px; border-radius: 12px; border: 1px solid rgba(30, 41, 59, 1);">
                <span style="color: #94A3B8; font-size: 11px;">OCR Text Similarity:</span>
                <div style="color: #10B981; font-weight: 800; font-size: 16px;">{ocr_pct}%</div>
            </div>
            <div style="background: rgba(2, 6, 23, 0.6); padding: 12px; border-radius: 12px; border: 1px solid rgba(30, 41, 59, 1);">
                <span style="color: #94A3B8; font-size: 11px;">Layout Structure Sim:</span>
                <div style="color: #6366F1; font-weight: 800; font-size: 16px;">{layout_pct}%</div>
            </div>
        </div>
    </div>
    """
