import streamlit as st
from src.core.provenance.registry import BlockchainProvenanceRegistry

def render_provenance_ui(document_id: str, file_path: str):
    """
    Renders a 'Verify Integrity' button to fetch the transaction from the blockchain
    and prove the document hash hasn't changed.
    """
    st.subheader("Document Provenance (Blockchain)")
    
    if st.button("Verify Integrity"):
        registry = BlockchainProvenanceRegistry()
        is_valid, message = registry.verify_integrity(document_id, file_path)
        
        if is_valid:
            st.success(message)
        else:
            st.error(message)
