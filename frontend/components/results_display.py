"""
Results display component for the Semantic Plagiarism Detector
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, List


class ResultsDisplay:
    """
    Component for displaying analysis results.
    """
    
    SEVERITY_COLORS = {
        'high': '#dc2626',
        'medium': '#d97706',
        'low': '#2563eb',
        'none': '#94a3b8'
    }
    
    SEVERITY_ICONS = {
        'high': '🔴',
        'medium': '🟡',
        'low': '🟢',
        'none': '⚪'
    }
    
    @classmethod
    def render(cls, results: Dict[str, Any]):
        """
        Render the results display component.
        
        Args:
            results: Analysis results dictionary
        """
        st.markdown("""
        <style>
            .result-summary {
                background: linear-gradient(135deg, #f8fafc, #eef2ff);
                border-radius: 12px;
                padding: 20px;
                margin-bottom: 20px;
                border: 1px solid #e2e8f0;
            }
            .result-stat {
                display: inline-block;
                padding: 4px 16px;
                border-radius: 20px;
                font-size: 14px;
                font-weight: 600;
            }
            .stat-total { background: #dbeafe; color: #1d4ed8; }
            .stat-high { background: #fee2e2; color: #dc2626; }
            .stat-medium { background: #fef3c7; color: #d97706; }
            .stat-low { background: #dcfce7; color: #16a34a; }
            .match-card {
                padding: 12px 16px;
                border-radius: 8px;
                margin-bottom: 8px;
                border-left: 4px solid #94a3b8;
                background: white;
                box-shadow: 0 1px 3px rgba(0,0,0,0.04);
            }
            .match-card.high { border-left-color: #dc2626; }
            .match-card.medium { border-left-color: #d97706; }
            .match-card.low { border-left-color: #2563eb; }
            .match-score {
                font-weight: 700;
                font-size: 18px;
            }
            .match-score.high { color: #dc2626; }
            .match-score.medium { color: #d97706; }
            .match-score.low { color: #2563eb; }
        </style>
        """, unsafe_allow_html=True)
        
        matches = results.get('matches', [])
        summary = results.get('summary', {})
        
        # Summary stats
        cls._render_summary(matches, summary)
        
        # Matches table
        cls._render_matches_table(matches)
        
        # Detailed view
        cls._render_detailed_matches(matches)
    
    @classmethod
    def _render_summary(cls, matches: List[Dict], summary: Dict):
        """Render summary statistics."""
        total = len(matches)
        high = sum(1 for m in matches if m.get('severity') == 'high')
        medium = sum(1 for m in matches if m.get('severity') == 'medium')
        low = sum(1 for m in matches if m.get('severity') == 'low')
        none = total - high - medium - low
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown(f"""
            <div class="result-summary" style="text-align:center;">
                <div style="font-size:28px;font-weight:700;">{total}</div>
                <div style="font-size:13px;color:#64748b;">Total Matches</div>
                <span class="result-stat stat-total">All</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="result-summary" style="text-align:center;border-color:#dc2626;">
                <div style="font-size:28px;font-weight:700;color:#dc2626;">{high}</div>
                <div style="font-size:13px;color:#64748b;">High Severity</div>
                <span class="result-stat stat-high">≥80%</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="result-summary" style="text-align:center;border-color:#d97706;">
                <div style="font-size:28px;font-weight:700;color:#d97706;">{medium}</div>
                <div style="font-size:13px;color:#64748b;">Medium Severity</div>
                <span class="result-stat stat-medium">60-80%</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="result-summary" style="text-align:center;border-color:#2563eb;">
                <div style="font-size:28px;font-weight:700;color:#2563eb;">{low}</div>
                <div style="font-size:13px;color:#64748b;">Low Severity</div>
                <span class="result-stat stat-low">40-60%</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            st.markdown(f"""
            <div class="result-summary" style="text-align:center;border-color:#94a3b8;">
                <div style="font-size:28px;font-weight:700;color:#94a3b8;">{none}</div>
                <div style="font-size:13px;color:#64748b;">No Match</div>
                <span class="result-stat" style="background:#e5e7eb;color:#4b5563;">{none}</span>
            </div>
            """, unsafe_allow_html=True)
    
    @classmethod
    def _render_matches_table(cls, matches: List[Dict]):
        """Render matches as a table."""
        if not matches:
            st.info("No matches found")
            return
        
        st.markdown("#### 📋 Match Details")
        
        # Prepare data for table
        data = []
        for match in matches:
            severity = match.get('severity', 'none')
            data.append({
                'Source': match.get('source', 'Unknown')[:30],
                'Target': match.get('target', 'Unknown')[:30],
                'Score': f"{match.get('score', 0):.2%}",
                'Severity': f"{cls.SEVERITY_ICONS.get(severity, '')} {severity.title()}",
                'Method': match.get('method', 'hybrid')
            })
        
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    @classmethod
    def _render_detailed_matches(cls, matches: List[Dict]):
        """Render detailed match cards."""
        if not matches:
            return
        
        st.markdown("#### 🔍 Detailed Matches")
        
        for match in matches[:20]:
            severity = match.get('severity', 'none')
            score = match.get('score', 0)
            source = match.get('source', 'Unknown')
            target = match.get('target', 'Unknown')
            
            severity_class = severity
            
            st.markdown(f"""
            <div class="match-card {severity_class}">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div>
                        <strong>📄 {source}</strong> → <strong>{target}</strong>
                    </div>
                    <div>
                        <span class="match-score {severity_class}">{score:.2%}</span>
                        <span style="margin-left:8px;">{cls.SEVERITY_ICONS.get(severity, '')}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)