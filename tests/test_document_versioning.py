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

import pytest

from src.core.document_versioning import DocumentDiffEngine
from src.db.version_history_db import VERSION_LINEAGE_CACHE, register_document_draft
from src.visualization.diff_heatmap import generate_evolution_heatmap


@pytest.fixture(autouse=True)
def clear_mock_db():
    VERSION_LINEAGE_CACHE.clear()


def test_diff_engine_identifies_additions_and_deletions():
    v1_text = "The system treats every upload as an independent profile framework."
    v2_text = "The system handles every upload as an independent draft blueprint."

    tokens = DocumentDiffEngine.compute_word_diff(v1_text, v2_text)
    metrics = DocumentDiffEngine.calculate_retention_metrics(tokens)

    actions = [t["action"] for t in tokens]

    assert "deleted" in actions
    assert "added" in actions
    assert metrics["retention_rate"] < 100.0


def test_version_lineage_increments_correctly():
    u_id = "user_student_abc"

    d1 = register_document_draft(
        u_id, "Initial thesis layout guidelines.", "draft1.txt"
    )
    d2 = register_document_draft(
        u_id, "Initial thesis layout modifications and details.", "draft2.txt"
    )

    assert d1["version_number"] == 1
    assert d2["version_number"] == 2
    assert d2["parent_hash"] == d1["doc_hash"]


def test_visualization_heatmap_generation_structure():
    mock_tokens = [{"text": "word", "action": "unchanged"}] * 60 + [
        {"text": "new", "action": "added"}
    ] * 10
    fig = generate_evolution_heatmap(mock_tokens, block_size=20)

    assert fig.data[0].type == "heatmap"
    assert len(fig.data[0].z[0]) == 4  # 70 tokens total / size 20 = 4 chunk blocks
