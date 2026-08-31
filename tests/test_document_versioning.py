import pytest
from src.core.document_versioning import DocumentDiffEngine
from src.db.version_history_db import register_document_draft, VERSION_LINEAGE_CACHE
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
    
    d1 = register_document_draft(u_id, "Initial thesis layout guidelines.", "draft1.txt")
    d2 = register_document_draft(u_id, "Initial thesis layout modifications and details.", "draft2.txt")
    
    assert d1["version_number"] == 1
    assert d2["version_number"] == 2
    assert d2["parent_hash"] == d1["doc_hash"]

def test_visualization_heatmap_generation_structure():
    mock_tokens = [{"text": "word", "action": "unchanged"}] * 60 + [{"text": "new", "action": "added"}] * 10
    fig = generate_evolution_heatmap(mock_tokens, block_size=20)
    
    assert fig.data[0].type == "heatmap"
    assert len(fig.data[0].z[0]) == 4 # 70 tokens total / size 20 = 4 chunk blocks
