import pytest
from datetime import datetime
from src.core.graph_engine.engine import (
    PlagiarismGraphEngine, DocumentNode, StudentNode, SimilarityEdge
)

@pytest.fixture
def graph_engine():
    return PlagiarismGraphEngine()

def test_node_addition(graph_engine):
    graph_engine.add_document("doc1", "Assignment 1")
    graph_engine.add_student("stu1", "John Doe")
    
    assert graph_engine.graph.number_of_nodes() == 2
    assert "Document" in graph_engine.graph.nodes["doc1"]["labels"]
    assert "Student" in graph_engine.graph.nodes["stu1"]["labels"]

def test_edge_addition_and_deduplication(graph_engine):
    # Test valid edge
    graph_engine.add_similarity_edge(SimilarityEdge("doc1", "doc2", 0.8, datetime.now()))
    
    # Should auto-create missing nodes
    assert graph_engine.graph.number_of_nodes() == 2
    assert graph_engine.graph.number_of_edges() == 1
    
    # Add duplicate edge - in networkx Graph this overwrites the edge attributes
    graph_engine.add_similarity_edge(SimilarityEdge("doc1", "doc2", 0.9, datetime.now()))
    assert graph_engine.graph.number_of_edges() == 1
    assert graph_engine.graph.edges["doc1", "doc2"]["score"] == 0.9

def test_invalid_edge_handling(graph_engine):
    # Empty dataset or self-loops
    graph_engine.add_similarity_edge(SimilarityEdge("doc1", "doc1", 0.8, datetime.now()))
    assert graph_engine.graph.number_of_edges() == 0
    
    # Invalid scores
    graph_engine.add_similarity_edge(SimilarityEdge("doc1", "doc2", 1.5, datetime.now()))
    assert graph_engine.graph.number_of_edges() == 0

def test_faiss_adapter(graph_engine):
    flags = [
        {"doc_a": "A", "doc_b": "B", "similarity": 0.85},
        {"doc_a": "B", "doc_b": "C", "similarity": 0.90},
        {"doc_a": "C", "doc_b": "D", "similarity": 0.30},
    ]
    graph_engine.build_from_faiss_flags(flags, default_year=2024)
    
    assert graph_engine.graph.number_of_nodes() == 4
    assert graph_engine.graph.number_of_edges() == 3

def test_filtering_and_thresholding(graph_engine):
    flags = [
        {"doc_a": "A", "doc_b": "B", "similarity": 0.85},
        {"doc_a": "B", "doc_b": "C", "similarity": 0.90},
        {"doc_a": "C", "doc_b": "D", "similarity": 0.30},
    ]
    graph_engine.build_from_faiss_flags(flags, default_year=2024)
    
    filtered = graph_engine.filter_graph(min_similarity=0.80)
    # Node D should be isolated and removed, leaving A, B, C
    assert filtered.number_of_nodes() == 3
    assert filtered.number_of_edges() == 2

def test_community_detection(graph_engine):
    # Create a dense triangle (suspicious ring)
    flags = [
        {"doc_a": "A", "doc_b": "B", "similarity": 0.95},
        {"doc_a": "B", "doc_b": "C", "similarity": 0.95},
        {"doc_a": "C", "doc_b": "A", "similarity": 0.95},
        # Create a separate minor component
        {"doc_a": "X", "doc_b": "Y", "similarity": 0.60},
    ]
    graph_engine.build_from_faiss_flags(flags)
    
    rings = graph_engine.detect_suspicious_rings(min_similarity=0.50)
    
    assert len(rings) == 2
    
    # Find the suspicious ring
    suspicious = [r for r in rings if r.is_suspicious]
    assert len(suspicious) == 1
    assert suspicious[0].size == 3
    assert suspicious[0].average_similarity == 0.95

def test_centrality_and_stats(graph_engine):
    # Star graph topology (Source copied by many)
    flags = [
        {"doc_a": "Source", "doc_b": "Copy1", "similarity": 0.90},
        {"doc_a": "Source", "doc_b": "Copy2", "similarity": 0.90},
        {"doc_a": "Source", "doc_b": "Copy3", "similarity": 0.90},
    ]
    graph_engine.build_from_faiss_flags(flags)
    
    stats = graph_engine.get_graph_stats()
    
    assert stats.node_count == 4
    assert stats.edge_count == 3
    assert stats.connected_components == 1
    
    # 'Source' should be the highest centrality node
    assert stats.highly_connected_students[0]['id'] == "Source"

def test_empty_dataset_handling(graph_engine):
    stats = graph_engine.get_graph_stats()
    assert stats.node_count == 0
    assert len(graph_engine.detect_suspicious_rings()) == 0
    assert graph_engine.calculate_centrality() == {}
