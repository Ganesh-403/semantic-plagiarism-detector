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

with open("src/db/incidents.py") as f:
    lines = f.readlines()

new_lines = []
for i, line in enumerate(lines):
    new_lines.append(line)
    if line.startswith("def build_incident_id"):
        pass

# I'll just write a script to insert it
insert_idx = -1
for i, line in enumerate(lines):
    if line.startswith("def _get_connection"):
        insert_idx = i
        break

func_str = """
def _parse_incident_id(val: str | int | None) -> int | None:
    if val is None:
        return None
    if isinstance(val, int):
        return val
    val_str = str(val).strip()
    if val_str.isdigit():
        return int(val_str)
    if val_str.startswith("INC-"):
        try:
            return int(val_str[4:], 16)
        except ValueError:
            pass
    return None

"""

lines.insert(insert_idx, func_str)

for i, line in enumerate(lines):
    if "MatchResult(" in line:
        # Check next few lines for incident_id=row["incident_id"]
        pass

with open("src/db/incidents.py", "w") as f:
    f.writelines(lines)
