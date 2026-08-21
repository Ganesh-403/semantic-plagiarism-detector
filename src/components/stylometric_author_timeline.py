"""Telemetry Ledger Component for Stylometric Author Attribution Records."""

from typing import Any, List


def render_stylometric_timeline(matches: List[Any]) -> str:
    """Renders HTML telemetry timeline list for stylometric author comparisons."""
    if not matches:
        return """
        <div style="text-align: center; padding: 40px; border: 1px dashed #334155; border-radius: 16px;">
            <p style="color: #94A3B8; font-size: 14px;">No stylometric author attributions logged in memory.</p>
        </div>
        """

    items_html = ""
    for match in matches:
        m_dict = match if isinstance(match, dict) else match.__dict__
        conf = m_dict.get("attribution_confidence_percentage", 0.0)
        is_same = m_dict.get("is_same_author", False)
        status_color = "#10B981" if is_same else "#6366F1"

        items_html += f"""
        <div style="
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(51, 65, 85, 0.8);
            border-radius: 14px;
            padding: 16px;
            margin-bottom: 12px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        ">
            <div>
                <span style="color: #6EE7B7; font-weight: 800; font-size: 14px;">
                    {m_dict.get("match_id", "STYLE-NODE")}
                </span>
                <div style="color: #94A3B8; font-size: 12px; margin-top: 4px;">
                    Author Candidate: {m_dict.get("candidate_author_alias")} | Trait: {m_dict.get("dominant_stylometric_trait")}
                </div>
            </div>
            <div style="text-align: right;">
                <span style="color: {status_color}; font-weight: 900; font-size: 16px;">
                    {conf}% Conf
                </span>
            </div>
        </div>
        """

    return f"""
    <div style="
        background: rgba(15, 23, 42, 0.9);
        border: 1px solid rgba(51, 65, 85, 1);
        border-radius: 20px;
        padding: 24px;
        margin-top: 24px;
    ">
        <h3 style="color: white; font-weight: 900; margin-top: 0; margin-bottom: 16px;">
            Stylometric Author Attribution Telemetry Ledger ({len(matches)} Scans Conducted)
        </h3>
        {items_html}
    </div>
    """
