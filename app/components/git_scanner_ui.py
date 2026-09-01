import streamlit as st
import pandas as pd
from typing import Dict, Any

def render_git_timeline_visualization(scan_results: Dict[str, Any]) -> None:
    """Render a timeline visualization of git commits and suspicious velocities."""
    st.subheader("Git Repository Commit Timeline")
    
    if not scan_results or "velocity" not in scan_results:
        st.info("No commit data available.")
        return
        
    velocities = scan_results["velocity"]
    if not velocities:
        st.info("Not enough commits to calculate velocity.")
        return
        
    df = pd.DataFrame(velocities)
    st.line_chart(df, y="lines_per_second")
    
    suspicious = scan_results.get("suspicious_commits", [])
    if suspicious:
        st.error(f"⚠️ Found {len(suspicious)} suspicious commit patterns!")
        st.dataframe(pd.DataFrame(suspicious))
    else:
        st.success("No suspicious incremental plagiarism detected.")

