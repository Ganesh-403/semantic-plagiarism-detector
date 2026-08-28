# MIT License
#
# Copyright (c) 2026 Ganesh Kambli
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Telemetry Ledger Component for FAISS Vector Search Audit Records."""

from typing import Any, List


def render_vector_search_timeline(reports: list[Any]) -> str:
    """Renders HTML telemetry timeline list for FAISS vector search queries."""
    if not reports:
        return """
        <div style="text-align: center; padding: 40px; border: 1px dashed #334155; border-radius: 16px;">
            <p style="color: #94A3B8; font-size: 14px;">No vector similarity search queries logged in memory.</p>
        </div>
        """

    items_html = ""
    for report in reports:
        rep_dict = report if isinstance(report, dict) else report.__dict__
        highest_pct = int(rep_dict.get("highest_similarity_ratio", 0.0) * 100)

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
                <span style="color: #10B981; font-weight: 800; font-size: 14px;">
                    {rep_dict.get('query_id', 'QRY-NODE')}
                </span>
                <div style="color: #94A3B8; font-size: 12px; margin-top: 4px;">
                    Query: "{rep_dict.get('query_text')[:60]}..." | Exec: {rep_dict.get('execution_time_ms')} ms
                </div>
            </div>
            <div style="text-align: right;">
                <span style="color: #F59E0B; font-weight: 900; font-size: 18px;">
                    {highest_pct}% Max Sim
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
            FAISS Vector Similarity Search Telemetry Ledger ({len(reports)} Queries Executed)
        </h3>
        {items_html}
    </div>
    """
