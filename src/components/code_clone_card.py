"""Streamlit Component for rendering Neural Code Clone Card metrics."""

from typing import Any, Dict


def render_code_clone_card(clone: Dict[str, Any]) -> str:
    """Generates HTML glassmorphic markup for displaying code clone details."""
    overall_pct = int(clone.get("overall_clone_score", 0.0) * 100)
    ast_pct = int(clone.get("ast_similarity_score", 0.0) * 100)
    semantic_pct = int(clone.get("neural_semantic_similarity", 0.0) * 100)

    badge_color = (
        "#10B981" if overall_pct < 40 else "#F59E0B" if overall_pct < 75 else "#EF4444"
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
                {clone.get("clone_type", "Type-3 Clone")}
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
                {overall_pct}% Clone Score
            </span>
        </div>
        
        <h4 style="color: white; font-weight: 900; margin: 0 0 8px 0;">
            Source: {clone.get("source_file_id")} vs Target: {clone.get("target_file_id")}
        </h4>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 16px;">
            <div style="background: rgba(2, 6, 23, 0.6); padding: 12px; border-radius: 12px; border: 1px solid rgba(30, 41, 59, 1);">
                <span style="color: #94A3B8; font-size: 11px;">AST Structural Sim:</span>
                <div style="color: #6366F1; font-weight: 800; font-size: 16px;">{ast_pct}%</div>
            </div>
            <div style="background: rgba(2, 6, 23, 0.6); padding: 12px; border-radius: 12px; border: 1px solid rgba(30, 41, 59, 1);">
                <span style="color: #94A3B8; font-size: 11px;">Neural Semantic Sim:</span>
                <div style="color: #10B981; font-weight: 800; font-size: 16px;">{semantic_pct}%</div>
            </div>
        </div>
    </div>
    """
