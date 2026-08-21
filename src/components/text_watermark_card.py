"""Streamlit Component for rendering Adversarial Watermark Cards."""

from typing import Any, Dict


def render_watermark_card(match: Dict[str, Any]) -> str:
    """Renders HTML glassmorphic markup for displaying watermark test metrics."""
    confidence = match.get("watermark_confidence_percentage", 0.0)
    z_score = match.get("z_score", 0.0)
    is_present = match.get("is_watermark_present", False)
    token_dist = match.get("token_distribution", {})
    if hasattr(token_dist, "__dict__"):
        token_dist = token_dist.__dict__

    badge_color = "#EF4444" if is_present else "#10B981"
    status_text = "WATERMARK DETECTED" if is_present else "UNWATERMARKED / HUMAN"

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
                background: rgba(245, 158, 11, 0.15);
                border: 1px solid rgba(245, 158, 11, 0.4);
                color: #FCD34D;
                font-size: 11px;
                font-weight: 800;
                padding: 4px 12px;
                border-radius: 9999px;
            ">
                Signature: {match.get("model_generator_signature")}
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
            Document: {match.get("document_title")} ({match.get("detection_id")})
        </h4>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; margin-top: 16px;">
            <div style="background: rgba(2, 6, 23, 0.6); padding: 12px; border-radius: 12px; border: 1px solid rgba(30, 41, 59, 1);">
                <span style="color: #94A3B8; font-size: 11px;">z-Score Test Statistic:</span>
                <div style="color: #F59E0B; font-weight: 800; font-size: 16px;">{z_score}</div>
            </div>
            <div style="background: rgba(2, 6, 23, 0.6); padding: 12px; border-radius: 12px; border: 1px solid rgba(30, 41, 59, 1);">
                <span style="color: #94A3B8; font-size: 11px;">Green-List Token Ratio:</span>
                <div style="color: #10B981; font-weight: 800; font-size: 16px;">{int(token_dist.get("observed_green_ratio", 0.0) * 100)}%</div>
            </div>
            <div style="background: rgba(2, 6, 23, 0.6); padding: 12px; border-radius: 12px; border: 1px solid rgba(30, 41, 59, 1);">
                <span style="color: #94A3B8; font-size: 11px;">p-Value Probability:</span>
                <div style="color: #6366F1; font-weight: 800; font-size: 16px;">{match.get("p_value")}</div>
            </div>
        </div>
    </div>
    """
