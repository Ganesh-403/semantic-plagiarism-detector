"""
tests/components/test_incident_export.py
-----------------------------------------
Unit tests for app.components.incident_export module.
"""

from unittest.mock import MagicMock, patch

from app.components.incident_export import render_incident_export_panel


@patch("app.components.incident_export.st")
@patch("app.components.incident_export.sync_flagged_incidents")
def test_render_incident_export_panel_empty(mock_sync, mock_st):
    """Test panel rendering when no incidents are present."""
    mock_sync.return_value = []
    render_incident_export_panel([])
    mock_st.info.assert_called_once_with(
        "No plagiarism incidents are currently available for export."
    )


@patch("app.components.incident_export.st")
@patch("app.components.incident_export.sync_flagged_incidents")
def test_render_incident_export_panel_with_copy_details(mock_sync, mock_st):
    """Test panel rendering and copy details code box generation."""
    # Return 3 mock columns for st.columns(3)
    mock_st.columns.return_value = (MagicMock(), MagicMock(), MagicMock())

    sample_incidents = [
        {
            "incident_id": "INC-001",
            "document_a": "doc1.pdf",
            "document_b": "doc2.pdf",
            "similarity_score": 0.85,
            "threshold_at_time_of_flag": 0.80,
            "severity_rank": "High",
            "review_status": "Pending",
            "date_flagged": "2026-08-01",
        }
    ]
    mock_sync.return_value = sample_incidents
    mock_st.selectbox.side_effect = ["All", "INC-001", "Pending"]
    
    render_incident_export_panel(sample_incidents)

    # Verify st.code was called with the expected formatted quick-copy summary string
    expected_summary = "Incident ID: #INC-001 | Similarity: 85.0% | Doc A: doc1.pdf | Doc B: doc2.pdf"
    mock_st.code.assert_called_once_with(expected_summary, language="text")
    