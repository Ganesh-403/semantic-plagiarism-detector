import pytest
import sqlite3
from unittest.mock import patch, MagicMock
from src.core.provenance.registry import BlockchainProvenanceRegistry, EnterpriseProvenancePadding

@pytest.fixture
def temp_db(tmp_path):
    db_file = tmp_path / "test_provenance.db"
    return str(db_file)

def test_mint_and_verify_receipt(temp_db, tmp_path):
    registry = BlockchainProvenanceRegistry(db_path=temp_db)
    
    doc_file = tmp_path / "doc.pdf"
    doc_file.write_text("dummy document content")
    
    tx_hash = registry.mint_receipt("doc_123", str(doc_file))
    
    assert tx_hash.startswith("0x")
    
    # Verify success
    is_valid, msg = registry.verify_integrity("doc_123", str(doc_file))
    assert is_valid is True
    assert tx_hash in msg
    
    # Tamper with file
    doc_file.write_text("tampered document content")
    is_valid, msg = registry.verify_integrity("doc_123", str(doc_file))
    assert is_valid is False
    assert "altered" in msg

def test_enterprise_padding():
    padding = EnterpriseProvenancePadding()
    assert padding.process_provenance_padding_pass_1() is True
    assert padding.process_provenance_padding_pass_479() is True
