"""Telemetry Ledger Component for Code Clone Audit Records."""

from typing import Any, List


def render_code_clone_timeline(matches: List[Any]) -> str:
    """Renders HTML telemetry timeline list for code clone detections."""
    if not matches:
        return """
        <div style="text-align: center; padding: 40px; border: 1px dashed #334155; border-radius: 16px;">
            <p style="color: #94A3B8; font-size: 14px;">No code clone audit matches logged in memory.</p>
        </div>
        """

    items_html = ""
    for match in matches:
        match_dict = match if isinstance(match, dict) else match.__dict__
        overall_pct = int(match_dict.get("overall_clone_score", 0.0) * 100)

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
                <span style="color: #6366F1; font-weight: 800; font-size: 14px;">
                    {match_dict.get("clone_id", "CLONE-NODE")}
                </span>
                <div style="color: #94A3B8; font-size: 12px; margin-top: 4px;">
                    {match_dict.get("source_file_id")} ↔ {match_dict.get("target_file_id")} ({match_dict.get("clone_type")})
                </div>
            </div>
            <div style="text-align: right;">
                <span style="color: #F59E0B; font-weight: 900; font-size: 18px;">
                    {overall_pct}% Match
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
            Neural Code Clone Telemetry Ledger ({len(matches)} Detections)
        </h3>
        {items_html}
    </div>
    """
