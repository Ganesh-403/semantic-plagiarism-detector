from pathlib import Path


SCRIPT_PATH = Path("scripts/generate_seed_data.py")


def test_script_exposes_target_similarity_flag():
    source = SCRIPT_PATH.read_text(encoding="utf-8")

    assert '"--target-similarity"' in source
    assert "type=parse_target_similarity" in source


def test_incident_uses_validated_similarity():
    source = SCRIPT_PATH.read_text(encoding="utf-8")

    assert (
        '"similarity": actual_target_similarity'
        in source
    )
    assert (
        "actual_target_similarity = "
        "validate_target_similarity("
        in source
    )


def test_default_similarity_remains_95_percent():
    source = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "DEFAULT_TARGET_SIMILARITY = 0.95" in source
