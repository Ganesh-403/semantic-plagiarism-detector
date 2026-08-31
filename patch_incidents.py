with open("src/db/incidents.py", "r") as f:
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
