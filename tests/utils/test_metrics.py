import json
import pytest

from src.utils.metrics import generate_metrics_json


def test_generate_metrics_json_structure() -> None:
    """Verify that generate_metrics_json produces valid JSON-serializable dictionaries containing expected keys."""
    metrics_data = generate_metrics_json()

    # 1. Assert return type is a dictionary
    assert isinstance(
        metrics_data, dict
    ), "generate_metrics_json() must return a dictionary"

    # 2. Assert dictionary can be serialized to JSON without raising a TypeError
    try:
        serialized = json.dumps(metrics_data)
    except TypeError as e:
        pytest.fail(f"generate_metrics_json() output is not JSON-serializable: {e}")

    # 3. Verify it deserializes back correctly and is non-empty
    deserialized = json.loads(serialized)
    assert isinstance(deserialized, dict)
    assert len(deserialized) > 0, "Metrics dictionary is empty"

    # 4. Assert presence of expected core metric keys (adjust keys as per your implementation)
    expected_keys = {"timestamp", "total_processed", "avg_similarity_score"}
    for key in expected_keys:
        assert (
            key in deserialized
        ), f"Expected key '{key}' missing from generate_metrics_json() output"
