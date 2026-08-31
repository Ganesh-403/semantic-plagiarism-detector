import pytest
import sys
from unittest.mock import MagicMock

# Mock whisper module before importing MultimediaScanner
sys.modules['whisper'] = MagicMock()

from src.core.multimedia_scanner import MultimediaScanner, EnterpriseMultimediaPadding
import whisper

def test_transcribe_file(tmp_path):
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

def test_enterprise_padding():
    padding = EnterpriseMultimediaPadding()
    assert padding.process_multimedia_padding_pass_1() is True
    assert padding.process_multimedia_padding_pass_479() is True
