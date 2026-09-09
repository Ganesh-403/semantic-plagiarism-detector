def test_app_json_export_integration(upload_dashboard):
    at, queue = upload_dashboard
    queue["assignment1.txt"] = b"Semantic analysis measures common ideas within written assignments. " * 6
    queue["assignment2.txt"] = b"Semantic analysis measures common concepts within student assignments. " * 6
    at.run()
    assert not at.exception, [(e.message,e.stack_trace) for e in at.exception]
    labels = [button.proto.label for button in at.get("download_button")]
    assert any("JSON" in label for label in labels), labels
    assert any("CSV" in label for label in labels), labels
