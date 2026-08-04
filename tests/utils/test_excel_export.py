import inspect
import io
from typing import Generator
import pandas as pd
from src.utils.excel_export import generate_csv_matrix_stream


def test_generate_csv_matrix_stream():
    # Setup test DataFrame
    data = {
        "DocA.txt": [1.0, 0.85, 0.12],
        "DocB.txt": [0.85, 1.0, 0.45],
        "DocC.txt": [0.12, 0.45, 1.0],
    }
    df = pd.DataFrame(data, index=["DocA.txt", "DocB.txt", "DocC.txt"])

    # Test 1: Return type is a Generator
    stream = generate_csv_matrix_stream(df)
    assert inspect.isgenerator(stream)

    # Test 2: Verify chunk output
    chunks = list(stream)
    assert len(chunks) == len(df) + 1  # 1 header row + 3 data rows

    # Verify header line
    assert chunks[0].strip() == "Document,DocA.txt,DocB.txt,DocC.txt"

    # Verify data lines
    assert chunks[1].strip() == "DocA.txt,1.0,0.85,0.12"
    assert chunks[2].strip() == "DocB.txt,0.85,1.0,0.45"
    assert chunks[3].strip() == "DocC.txt,0.12,0.45,1.0"

    # Test 3: Verify complete CSV reconstruction matches Expected CSV output
    full_csv = "".join(chunks)
    reconstructed_df = pd.read_csv(io.StringIO(full_csv), index_col=0)
    pd.testing.assert_frame_equal(df, reconstructed_df)
