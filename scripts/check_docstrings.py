#!/usr/bin/env python3

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

import ast
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
COVERAGE_THRESHOLD = 85


def find_python_files():
    """Return all Python source files under src/."""

    python_files = []

    for file in SRC_DIR.rglob("*.py"):
        if "__pycache__" in file.parts:
            continue
        python_files.append(file)

    return python_files


def check_file(file_path):
    """Inspect a Python file and return docstring statistics."""

    try:
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError):
        return 0, 0, []

    total_functions = 0
    documented_functions = 0
    missing_functions = []

    for node in tree.body:
        # Top-level functions
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("_"):
                continue

            total_functions += 1

            if ast.get_docstring(node):
                documented_functions += 1
            else:
                missing_functions.append(node.name)

        # Methods inside classes
        elif isinstance(node, ast.ClassDef):
            for method in node.body:
                if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if method.name.startswith("_"):
                        continue

                    total_functions += 1

                    if ast.get_docstring(method):
                        documented_functions += 1
                    else:
                        missing_functions.append(f"{node.name}.{method.name}")

    return total_functions, documented_functions, missing_functions


def calculate_coverage(total_functions, documented_functions):
    """Calculate docstring coverage percentage."""

    if total_functions == 0:
        return 100.0

    return (documented_functions / total_functions) * 100


def main():
    files = find_python_files()

    total_functions = 0
    documented_functions = 0

    all_missing = []

    for file in files:
        total, documented, missing = check_file(file)

        total_functions += total
        documented_functions += documented

        if missing:
            all_missing.append((file, missing))

    coverage = calculate_coverage(
        total_functions,
        documented_functions,
    )

    if all_missing:
        print("Missing docstrings:\n")

        for file, functions in all_missing:
            print(file.relative_to(SRC_DIR.parent).as_posix())

            for function in functions:
                print(f"  - {function}")

            print()

    print(f"Docstring coverage: {coverage:.2f}%")
    print(f"Documented {documented_functions} of {total_functions} public functions.")

    if coverage >= COVERAGE_THRESHOLD:
        sys.exit(0)

    sys.exit(1)


if __name__ == "__main__":
    main()
