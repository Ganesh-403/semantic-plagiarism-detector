import streamlit as st

def render_multimedia_player(file_path: str, transcript_data: dict, plagiarized_segments: list):
    """
    Renders a custom Streamlit audio/video player.
    Highlights exact timestamps of plagiarized segments.
    """
    st.subheader("Multimedia Plagiarism Analysis")
    
    # Simple media player
    if file_path.endswith('.mp4'):
        st.video(file_path)
    else:
        st.audio(file_path)
        
    st.markdown("### Plagiarized Segments (Timestamps)")
    for segment in plagiarized_segments:
        start = segment.get("start", 0)
        end = segment.get("end", 0)
        text = segment.get("text", "")
        
        st.error(f"**[{start:.2f}s - {end:.2f}s]**: {text}")
