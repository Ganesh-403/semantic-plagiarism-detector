"""
API client for the Semantic Plagiarism Detector
"""

import requests
import json
from typing import Dict, Any, List, Optional
import streamlit as st


class APIClient:
    """
    Client for interacting with the backend API.
    """
    
    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or "http://localhost:8000"
        self.session = requests.Session()
    
    def analyze_documents(
        self,
        documents: List[Dict[str, Any]],
        method: str = "hybrid",
        threshold: float = 0.59
    ) -> Dict[str, Any]:
        """
        Send documents for analysis.
        
        Args:
            documents: List of document dictionaries
            method: Analysis method (hybrid, lexical, semantic)
            threshold: Similarity threshold
        
        Returns:
            Analysis results
        """
        try:
            response = self.session.post(
                f"{self.base_url}/api/analyze",
                json={
                    'documents': documents,
                    'method': method,
                    'threshold': threshold
                },
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    'error': f"API Error: {response.status_code}",
                    'matches': [],
                    'summary': {}
                }
        except requests.exceptions.ConnectionError:
            return {
                'error': "Cannot connect to backend server",
                'matches': [],
                'summary': {}
            }
        except requests.exceptions.Timeout:
            return {
                'error': "Request timed out",
                'matches': [],
                'summary': {}
            }
        except Exception as e:
            return {
                'error': str(e),
                'matches': [],
                'summary': {}
            }
    
    def check_health(self) -> bool:
        """
        Check if the backend is healthy.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            response = self.session.get(
                f"{self.base_url}/health",
                timeout=5
            )
            return response.status_code == 200
        except:
            return False