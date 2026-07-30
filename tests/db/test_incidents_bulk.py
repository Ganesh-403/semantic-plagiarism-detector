import sqlite3

import pytest

from src.db.corpus_db import clear_all_data
from src.db.incidents import (_fetch_all_incidents, init_incident_db,
                              sync_flagged_incidents)


@pytest.fixture(autouse=True)
def setup_teardown():
    init_incident_db()
    clear_all_data()
    yield
    clear_all_data()

def test_sync_flagged_incidents_bulk():
    flags = [
        {"doc_a": "doc1.pdf", "doc_b": "doc2.pdf", "similarity": 0.85, "severity": "High"},
        {"doc_a": "doc2.pdf", "doc_b": "doc3.pdf", "similarity": 0.45, "severity": "Low"},
        {"doc_a": "doc1.pdf", "doc_b": "doc3.pdf", "similarity": 0.95, "severity": "Critical"}
    ]
    
    # Bulk insert via executemany implementation
    sync_flagged_incidents(flags)
    
    # Verify records in database
    conn = sqlite3.connect(r"corpus.db")
    incidents = _fetch_all_incidents(conn)
    conn.close()
    
    assert len(incidents) == 3
    
    # Check if similarity scores correctly inserted
    scores = [inc["similarity_score"] for inc in incidents]
    assert 0.85 in scores
    assert 0.45 in scores
    assert 0.95 in scores
    
def test_sync_flagged_incidents_bulk_upsert():
    flags = [
        {"doc_a": "doc1.pdf", "doc_b": "doc2.pdf", "similarity": 0.50, "severity": "Medium"}
    ]
    sync_flagged_incidents(flags)
    
    conn = sqlite3.connect(r"corpus.db")
    assert len(_fetch_all_incidents(conn)) == 1
    conn.close()
    
    # Update the existing record with new similarity
    flags_update = [
        {"doc_a": "doc1.pdf", "doc_b": "doc2.pdf", "similarity": 0.99, "severity": "Critical"}
    ]
    sync_flagged_incidents(flags_update)
    
    conn = sqlite3.connect(r"corpus.db")
    incidents = _fetch_all_incidents(conn)
    conn.close()
    
    assert len(incidents) == 1
    assert incidents[0]["similarity_score"] == 0.99
    assert incidents[0]["severity_rank"] == "High"

def test_sync_flagged_incidents_bulk_invalid():
    # Test skipping invalid pairs
    flags = [
        {"doc_a": "doc1.pdf", "doc_b": "doc1.pdf", "similarity": 1.0}, # Same doc
        {"doc_a": "", "doc_b": "doc2.pdf", "similarity": 0.8} # Missing doc A
    ]
    sync_flagged_incidents(flags)
    
    conn = sqlite3.connect(r"corpus.db")
    incidents = _fetch_all_incidents(conn)
    conn.close()
    
    assert len(incidents) == 0
