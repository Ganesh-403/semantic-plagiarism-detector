with open("src/core/similarity.py", "r") as f:
    text = f.read()

text = text.replace(
    "find_optimal_threshold)                                            find_optimal_threshold)",
    "find_optimal_threshold)"
)
text = text.replace(
    ") -> list[dict]:    \"\"\"Identify document pairs",
    ") -> list[dict]:\n    \"\"\"Identify document pairs"
)
text = text.replace(
    "if is_plagiarism(score, effective_threshold):            doc_b = doc_names[j]",
    "if is_plagiarism(score, effective_threshold):\n            doc_a = doc_names[i]\n            doc_b = doc_names[j]"
)
with open("src/core/similarity.py", "w") as f:
    f.write(text)

with open("src/core/similarity.py", "r") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if 885 <= i <= 950:
        if line.startswith("        flag_dict = {"):
            pass # We need to indent everything until the next outdent
        if not line.startswith("            ") and line.startswith("        "):
            lines[i] = "    " + line
        elif not line.startswith("            ") and line.startswith(" "):
            pass # Could be closing braces
            
with open("src/core/similarity.py", "w") as f:
    f.writelines(lines)
