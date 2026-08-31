from app.theme import pipeline_progress_html


def test_pipeline_progress_html_without_eta():
    result = pipeline_progress_html(["Extract", "Chunk"])

    assert "Extract" in result
    assert "Chunk" in result
    assert "Estimated processing time" not in result


def test_pipeline_progress_html_displays_eta_under_indicator():
    result = pipeline_progress_html(
        ["Extract", "Chunk", "Embed"],
        active_index=1,
        estimated_seconds=75,
    )

    assert 'class="pipeline-steps"' in result
    assert 'class="pipeline-eta"' in result
    assert ("Estimated processing time: about 1 minute 15 seconds") in result
    assert result.index("pipeline-steps") < result.index("pipeline-eta")
