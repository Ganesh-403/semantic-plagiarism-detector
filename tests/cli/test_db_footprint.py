import json
import pytest
from unittest.mock import patch

from src.cli import main

def test_db_footprint_text_output(capsys, monkeypatch):
    monkeypatch.setattr('sys.argv', ['cli.py', 'db', 'footprint'])

    mock_res = {
        'embedding_bytes': 1536,
        'database_bytes': 10240,
        'embedding_percentage': 15.0,
        'chunk_count': 2
    }

    with patch('src.db.corpus_db.get_embedding_storage_footprint', return_value=mock_res):
        with pytest.raises(SystemExit) as e:
            main()

        assert e.value.code == 0
        
    out, err = capsys.readouterr()
    assert 'Total Database Size: 10,240 bytes' in out
    assert 'Total Embedding Size: 1,536 bytes' in out
    assert 'Embedding Storage Percentage: 15.00%' in out
    assert 'Total Chunks: 2' in out


def test_db_footprint_json_output(capsys, monkeypatch):
    monkeypatch.setattr('sys.argv', ['cli.py', 'db', 'footprint', '--output-format', 'json'])

    mock_res = {
        'embedding_bytes': 4096,
        'database_bytes': 8192,
        'embedding_percentage': 50.0,
        'chunk_count': 5
    }

    with patch('src.db.corpus_db.get_embedding_storage_footprint', return_value=mock_res):
        with pytest.raises(SystemExit) as e:
            main()
            
        assert e.value.code == 0
        
    out, err = capsys.readouterr()
    parsed = json.loads(out)
    assert parsed['embedding_bytes'] == 4096
    assert parsed['embedding_percentage'] == 50.0


def test_db_footprint_error_handling(capsys, monkeypatch):
    monkeypatch.setattr('sys.argv', ['cli.py', 'db', 'footprint'])

    with patch('src.db.corpus_db.get_embedding_storage_footprint', side_effect=Exception('DB Error')):
        with pytest.raises(SystemExit) as e:
            main()
            
        assert e.value.code == 1
        
    out, err = capsys.readouterr()
    assert 'Error calculating storage footprint: DB Error' in err

