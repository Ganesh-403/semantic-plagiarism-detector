"""
File uploader component for the Semantic Plagiarism Detector
"""

import streamlit as st
import tempfile
import os
from pathlib import Path
from typing import List, Dict, Any, Optional


class FileUploader:
    """
    Component for uploading and processing documents.
    Supports multiple file formats.
    """
    
    SUPPORTED_TYPES = {
        'pdf': 'application/pdf',
        'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'doc': 'application/msword',
        'txt': 'text/plain',
        'rtf': 'application/rtf',
        'odt': 'application/vnd.oasis.opendocument.text'
    }
    
    SUPPORTED_EXTENSIONS = ['.pdf', '.docx', '.doc', '.txt', '.rtf', '.odt']
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    
    @classmethod
    def render(cls) -> List[Dict[str, Any]]:
        """
        Render the file uploader component.
        
        Returns:
            List of uploaded document dictionaries
        """
        st.markdown("""
        <style>
            .upload-area {
                border: 2px dashed #cbd5e1;
                border-radius: 12px;
                padding: 30px;
                text-align: center;
                background: #f8fafc;
                transition: all 0.3s ease;
                margin-bottom: 16px;
            }
            .upload-area:hover {
                border-color: #3b82f6;
                background: #eff6ff;
            }
            .upload-icon {
                font-size: 48px;
                margin-bottom: 8px;
            }
            .upload-text {
                font-size: 16px;
                color: #64748b;
            }
            .upload-subtext {
                font-size: 13px;
                color: #94a3b8;
                margin-top: 4px;
            }
            .file-list {
                margin-top: 12px;
            }
            .file-item {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 8px 12px;
                background: white;
                border-radius: 6px;
                margin-bottom: 6px;
                border: 1px solid #e2e8f0;
            }
            .file-name {
                font-weight: 500;
                color: #1e293b;
            }
            .file-size {
                font-size: 12px;
                color: #94a3b8;
            }
            .file-status {
                font-size: 12px;
                padding: 2px 10px;
                border-radius: 12px;
            }
            .file-status.success {
                background: #dcfce7;
                color: #16a34a;
            }
            .file-status.error {
                background: #fee2e2;
                color: #dc2626;
            }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="upload-area">
            <div class="upload-icon">📤</div>
            <div class="upload-text">Drop your files here or click to browse</div>
            <div class="upload-subtext">Supported: PDF, DOCX, DOC, TXT, RTF, ODT (Max 10MB)</div>
        </div>
        """, unsafe_allow_html=True)
        
        # File uploader
        uploaded_files = st.file_uploader(
            label="Upload documents",
            type=['pdf', 'docx', 'doc', 'txt', 'rtf', 'odt'],
            accept_multiple_files=True,
            key="file_uploader_widget",
            label_visibility="collapsed"
        )
        
        uploaded_docs = []
        
        if uploaded_files:
            st.markdown("#### 📄 Uploaded Files")
            
            for file in uploaded_files:
                # Validate file
                is_valid, error = cls._validate_file(file)
                
                if is_valid:
                    # Add to session state
                    doc = {
                        'name': file.name,
                        'content': None,
                        'size': file.size,
                        'type': Path(file.name).suffix[1:],
                        'file': file
                    }
                    
                    uploaded_docs.append(doc)
                    
                    # Display file
                    st.markdown(f"""
                    <div class="file-item">
                        <div>
                            <span class="file-name">📄 {file.name}</span>
                            <span class="file-size">({file.size / 1024:.1f} KB)</span>
                        </div>
                        <span class="file-status success">✅ Ready</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error(f"❌ {file.name}: {error}")
        
        return uploaded_docs
    
    @classmethod
    def _validate_file(cls, file) -> tuple:
        """
        Validate uploaded file.
        
        Args:
            file: Uploaded file object
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check file size
        if file.size > cls.MAX_FILE_SIZE:
            return False, f"File size exceeds {cls.MAX_FILE_SIZE // (1024*1024)}MB limit"
        
        # Check file extension
        ext = Path(file.name).suffix.lower()
        if ext not in cls.SUPPORTED_EXTENSIONS:
            return False, f"Unsupported file type: {ext}"
        
        return True, ""
    
    @classmethod
    def get_supported_formats(cls) -> List[str]:
        """Get list of supported file formats."""
        return cls.SUPPORTED_EXTENSIONS
    
    @classmethod
    def cleanup_temp_files(cls):
        """Clean up temporary files."""
        for doc in st.session_state.documents:
            if 'path' in doc and doc['path'] and os.path.exists(doc['path']):
                try:
                    os.unlink(doc['path'])
                except:
                    pass