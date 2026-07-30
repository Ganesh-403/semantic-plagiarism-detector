import io
import json
import zipfile

from src.utils.bulk_export import generate_bulk_reports_zip


def test_generate_bulk_reports_zip():
    flags = [
        {"doc1": "Alice.pdf", "doc2": "Bob.docx", "similarity_score": 0.85, "matched_chunks": []},
        {"doc1": "Charlie.txt", "doc2": "Dave.pdf", "similarity_score": 0.95, "matched_chunks": ["chunk1"]}
    ]
    
    zip_bytes = generate_bulk_reports_zip(flags)
    assert isinstance(zip_bytes, bytes)
    
    # Read the zip file
    with zipfile.ZipFile(io.BytesIO(zip_bytes), 'r') as zf:
        names = zf.namelist()
        assert len(names) == 2
        assert any("Alice" in name and "Bob" in name for name in names)
        assert any("Charlie" in name and "Dave" in name for name in names)
        
        # Check content
        first_file = names[0]
        content = zf.read(first_file).decode('utf-8')
        data = json.loads(content)
        assert "similarity_score" in data
        assert "generated_at" in data


def test_generate_bulk_reports_zip_with_progress_bar():
    from unittest.mock import Mock
    flags = [
        {"doc1": "Alice.pdf", "doc2": "Bob.docx", "similarity_score": 0.85, "matched_chunks": []},
        {"doc1": "Charlie.txt", "doc2": "Dave.pdf", "similarity_score": 0.95, "matched_chunks": ["chunk1"]}
    ]
    mock_pb = Mock()
    generate_bulk_reports_zip(flags, progress_bar=mock_pb)
    assert mock_pb.progress.call_count == 4
    # Verify the final state was reported
    mock_pb.progress.assert_any_call(1.0, text="ZIP archive ready!")
