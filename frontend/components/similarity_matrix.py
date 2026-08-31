"""
Similarity matrix component for the Semantic Plagiarism Detector
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, List, Optional
import numpy as np


class SimilarityMatrix:
    """
    Component for displaying similarity matrix with heatmap visualization.
    """
    
    @classmethod
    def render(cls, results: Dict[str, Any]):
        """
        Render the similarity matrix component.
        
        Args:
            results: Analysis results dictionary
        """
        st.markdown("""
        <style>
            .matrix-container {
                background: white;
                border-radius: 12px;
                padding: 20px;
                border: 1px solid #e2e8f0;
                margin-bottom: 16px;
            }
            .matrix-title {
                font-size: 18px;
                font-weight: 600;
                color: #1e293b;
                margin-bottom: 12px;
            }
        </style>
        """, unsafe_allow_html=True)
        
        matches = results.get('matches', [])
        documents = results.get('documents', [])
        
        if not matches:
            st.info("No data available for similarity matrix")
            return
        
        # Get document names
        doc_names = []
        for m in matches:
            if m.get('source') and m.get('source') not in doc_names:
                doc_names.append(m.get('source'))
            if m.get('target') and m.get('target') not in doc_names:
                doc_names.append(m.get('target'))
        
        # Build matrix
        n = len(doc_names)
        matrix_data = np.zeros((n, n))
        
        for match in matches:
            src = match.get('source', '')
            tgt = match.get('target', '')
            score = match.get('score', 0)
            
            if src in doc_names and tgt in doc_names:
                i = doc_names.index(src)
                j = doc_names.index(tgt)
                matrix_data[i][j] = score
                matrix_data[j][i] = score
        
        # Heatmap
        fig = go.Figure(data=go.Heatmap(
            z=matrix_data,
            x=doc_names,
            y=doc_names,
            colorscale='RdYlGn',
            zmin=0,
            zmax=1,
            text=matrix_data.round(2),
            texttemplate='%{text:.2f}',
            textfont={"size": 10},
            hoverongaps=False,
            hovertemplate='<b>%{x}</b> → <b>%{y}</b><br>Similarity: %{z:.2%}<extra></extra>'
        ))
        
        fig.update_layout(
            title={
                'text': '📊 Similarity Matrix',
                'font': {'size': 18, 'color': '#1e293b'}
            },
            height=500,
            xaxis={
                'title': 'Target Documents',
                'tickangle': 45,
                'tickfont': {'size': 10}
            },
            yaxis={
                'title': 'Source Documents',
                'tickfont': {'size': 10}
            },
            margin={'l': 40, 'r': 40, 't': 60, 'b': 80},
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Download button
        df = pd.DataFrame(matrix_data, index=doc_names, columns=doc_names)
        csv = df.to_csv()
        st.download_button(
            label="📥 Download Matrix as CSV",
            data=csv,
            file_name="similarity_matrix.csv",
            mime="text/csv",
            use_container_width=True
        )