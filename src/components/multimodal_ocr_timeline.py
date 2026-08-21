"""Telemetry Ledger Component for Multimodal Image OCR Audit Records."""

from typing import Any, List


def render_ocr_timeline(matches: List[Any]) -> str:
    """Renders HTML telemetry timeline list for OCR document image scans."""
    if not matches:
        return """
        <div style="text-align: center; padding: 40px; border: 1px dashed #334155; border-radius: 16px;">
            <p style="color: #94A3B8; font-size: 14px;">No image OCR document plagiarism scans logged in memory.</p>
        </div>
        """

    items_html = ""
    for match in matches:
        m_dict = match if isinstance(match, dict) else match.__dict__
        overall_pct = int(m_dict.get("overall_multimodal_score", 0.0) * 100)
        status_color = (
            "#EF4444"
            if overall_pct > 75
            else "#F59E0B"
            if overall_pct > 40
            else "#10B981"
        )

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
                    {m_dict.get("scan_id", "OCR-NODE")}
                </span>
                <div style="color: #94A3B8; font-size: 12px; margin-top: 4px;">
                    Image: {m_dict.get("source_image_name")} | Engine: {m_dict.get("ocr_engine_used")}
                </div>
            </div>
            <div style="text-align: right;">
                <span style="color: {status_color}; font-weight: 900; font-size: 16px;">
                    {overall_pct}% Score
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
            Multimodal Image OCR Telemetry Ledger ({len(matches)} Scans Conducted)
        </h3>
        {items_html}
    </div>
    """
