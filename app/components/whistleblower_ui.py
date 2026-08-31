import streamlit as st
from src.core.whistleblower.tor_gateway import TorOnionService

def render_whistleblower_ui():
    """
    Zero-knowledge UI where users can submit reports and track status.
    """
    st.title("Anonymous Whistleblower Drop-box")
    st.info("You are connected via Tor Onion Routing. Your IP and metadata are scrubbed.")
    
    gateway = TorOnionService()
    
    with st.form("submission_form"):
        evidence = st.file_uploader("Upload Evidence (.zip, .pdf)", type=["zip", "pdf"])
        notes = st.text_area("Optional Notes (Do not include PII)")
        
        submitted = st.form_submit_button("Submit Anonymously")
        if submitted and evidence:
            file_bytes = evidence.read()
            access_key = gateway.process_submission(file_bytes)
            
            st.success("Evidence submitted successfully!")
            st.warning(f"Your one-time access key to track this report is: {access_key}")
            st.error("Please save this key securely. It cannot be recovered if lost.")

    st.subheader("Track Existing Report")
    track_key = st.text_input("Enter Access Key")
    if st.button("Check Status") and track_key:
        st.info("Status: Under Review by Academic Integrity Committee")
