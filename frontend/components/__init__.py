"""
Frontend components package for the Semantic Plagiarism Detector
"""

from .file_uploader import FileUploader
from .results_display import ResultsDisplay
from .similarity_matrix import SimilarityMatrix
from .analysis_controls import AnalysisControls

__all__ = [
    'FileUploader',
    'ResultsDisplay',
    'SimilarityMatrix',
    'AnalysisControls'
]