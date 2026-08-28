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

import csv
import io
from datetime import datetime

# Software version constant
VERSION = "1.4.2"


def export_to_csv(results, threshold=0.8, include_metadata=True):
    """
    Exports plagiarism scan results to a CSV string with optional metadata headers.
    """
    output = io.StringIO()

    if include_metadata:
        scan_date = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        output.write(f"# Plagiarism Report\n")
        output.write(f"# Scan Date: {scan_date}\n")
        output.write(f"# Threshold Used: {threshold}\n")
        output.write(f"# Software Version: {VERSION}\n")
        output.write(f"#\n")  # Blank separator comment line

    writer = csv.writer(output)

    # Standard CSV column headers
    writer.writerow(["doc_a", "doc_b", "similarity_score"])

    # Data rows
    for row in results:
        writer.writerow(
            [row.get("doc_a"), row.get("doc_b"), row.get("similarity_score")]
        )

    return output.getvalue()
