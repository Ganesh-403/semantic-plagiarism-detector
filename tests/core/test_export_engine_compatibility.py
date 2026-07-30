import json

from src.core.export_engine import LMSExportEngine


INCIDENTS = [
    {
        "doc_a": "alpha.pdf",
        "doc_b": "beta.pdf",
        "similarity": 0.93,
    }
]


def test_csv_generation_api_is_preserved():
    result = LMSExportEngine.generate_incident_csv(INCIDENTS)

    assert result is not None
    assert "alpha.pdf" in result
    assert "CRITICAL" in result


def test_json_generation_api_is_preserved():
    result = LMSExportEngine.generate_incident_json(INCIDENTS)

    assert result is not None
    payload = json.loads(result)
    assert payload["metadata"]["total_incidents"] == 1


def test_txt_generation_api_is_preserved():
    result = LMSExportEngine.generate_incident_txt(INCIDENTS)

    assert result is not None
    assert "Similarity: 93.0%" in result


def test_empty_exports_still_return_none():
    assert LMSExportEngine.generate_incident_csv([]) is None
    assert LMSExportEngine.generate_incident_json([]) is None
    assert LMSExportEngine.generate_incident_txt([]) is None
