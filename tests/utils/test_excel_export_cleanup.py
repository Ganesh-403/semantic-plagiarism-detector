# MIT License
#
# Copyright (c) 2026 Ganesh Kambli
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import subprocess
import sys
from pathlib import Path


def test_excel_export_temp_file_cleaned_on_exit():
    """Temp .xlsx from export_similarity_matrix_to_temp_file is unlinked after exit."""
    root = Path(__file__).resolve().parents[2]
    child = f"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path({str(root)!r}) / "src" / "utils"))

import pandas as pd
import excel_export

df = pd.DataFrame(
    [[1.0, 0.4], [0.4, 1.0]],
    index=["a.txt", "b.txt"],
    columns=["a.txt", "b.txt"],
)
path = excel_export.export_similarity_matrix_to_temp_file(df)
assert os.path.isfile(path)
print(path)
"""
    result = subprocess.run(
        [sys.executable, "-c", child],
        cwd=str(root),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    temp_path = result.stdout.strip().splitlines()[-1]
    assert temp_path.endswith(".xlsx")
    assert not os.path.exists(temp_path)
