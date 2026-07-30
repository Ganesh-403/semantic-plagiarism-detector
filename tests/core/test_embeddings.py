
def test_embed_empty_text():
    vec = embed("")
    assert len(vec) == 0 or all(v == 0 for v in vec)
