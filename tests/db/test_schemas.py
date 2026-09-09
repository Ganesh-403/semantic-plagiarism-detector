from src.db.schemas import MatchResult


def test_match_result_integer_incident_id():
    # Test that integer incident_id round-trips correctly
    result = MatchResult(
        incident_id=123, document_a="a.txt", document_b="b.txt", similarity_score=0.9
    )
    assert result.incident_id == 123
    assert isinstance(result.incident_id, int)


def test_match_result_string_incident_id_conversion():
    # Test that string integer incident_id preserves the supplied identifier
    result = MatchResult(
        incident_id="123", document_a="a.txt", document_b="b.txt", similarity_score=0.9
    )
    assert result.incident_id == "123"
    assert isinstance(result.incident_id, str)
