import pytest
import sys
from unittest.mock import MagicMock

from src.core.multimedia_scanner import MultimediaScanner

def test_transcribe_file(tmp_path, monkeypatch):
    whisper = MagicMock()
    monkeypatch.setitem(sys.modules, "whisper", whisper)
    mock_model = MagicMock()
    mock_model.transcribe.return_value = {"text": "test", "segments": []}
    whisper.load_model.return_value = mock_model
    
    scanner = MultimediaScanner()
    
    test_file = tmp_path / "test.mp3"
    test_file.write_text("dummy")
    
    result = scanner.transcribe_file(str(test_file))
    
    whisper.load_model.assert_called_once_with("base")
    mock_model.transcribe.assert_called_once_with(str(test_file), word_timestamps=True)
    assert result["text"] == "test"



def test_missing_media_does_not_load_model(tmp_path, monkeypatch):
    scanner = MultimediaScanner()
    load = MagicMock()
    monkeypatch.setattr(scanner, "load_model", load)
    with pytest.raises(FileNotFoundError):
        scanner.transcribe_file(str(tmp_path / "missing.mp3"))
    load.assert_not_called()
