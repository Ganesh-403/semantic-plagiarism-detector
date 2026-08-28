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

with open("src/utils/bulk_export.py", "r") as f:
    content = f.read()

import_re = "import re\n"
if "import re" not in content:
    content = content.replace("import csv\n", "import csv\nimport re\n")

sanitize_func = """
def sanitize_export_filename(filename: str, default_ext: str = ".csv") -> str:
    \"\"\"
    Strip illegal OS/filesystem characters from the filename and ensure it ends with default_ext.
    \"\"\"
    sanitized = re.sub(r'[<>:"/\\\\|?*]', '', filename)
    if not sanitized.endswith(default_ext):
        sanitized += default_ext
    return sanitized

def export_incidents_csv(
    incidents_list: List[Dict],
    filename: str = "export.csv",
    delimiter: str = ",",
    quoting_style: int = csv.QUOTE_MINIMAL,
) -> Tuple[bytes, str]:
    \"\"\"Export a list of incident dicts to a CSV-formatted byte stream and a sanitized filename.

    Validates that the delimiter is a single character string, falling back to a
    comma if an invalid delimiter is supplied.
    \"\"\"
    if not isinstance(delimiter, str) or len(delimiter) != 1:
        delimiter = ","
    sanitized_filename = sanitize_export_filename(filename, ".csv")
    csv_bytes = export_incidents_csv_stream(
        incidents_list, delimiter=delimiter, quoting_style=quoting_style
    )
    return csv_bytes, sanitized_filename
"""

import re

content = re.sub(r"def export_incidents_csv\([\s\S]*?    \)", sanitize_func, content)
# wait, my regex above would delete the whole function body? No, just until the closing paren.
# let me replace the whole function.
