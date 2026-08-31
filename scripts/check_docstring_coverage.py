#!/usr/bin/env python3
"""
Docstring Coverage Checker Script

This utility script inspects module, class, and function docstrings in the `src/`
directory using Python's built-in `ast` module. It reports the documentation
coverage percentage and exits with code 0 if coverage is >= 80%, otherwise 1.

Usage:
    python scripts/check_docstring_coverage.py
"""

import ast
import sys
from pathlib import Path
from typing import Any, List, Tuple


class DocstringVisitor(ast.NodeVisitor):
    """AST Visitor to count documented modules, classes, and functions."""

    def __init__(self, filename: str):
        self.filename = filename
        self.total_nodes = 0
        self.documented_nodes = 0
        self.undocumented_items: list[str] = []

    def _check_docstring(self, node: Any, node_type: str) -> None:
        """Helper to check if a node has a docstring."""
        self.total_nodes += 1
        docstring = ast.get_docstring(node)

        if docstring and len(docstring.strip()) > 0:
            self.documented_nodes += 1
        else:
            name = getattr(node, "name", "<module>")
            self.undocumented_items.append(f"{node_type}: {name} in {self.filename}")

    def visit_Module(self, node: ast.Module) -> None:
        """Check module-level docstring."""
        self._check_docstring(node, "Module")
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Check class docstring."""
        self._check_docstring(node, "Class")
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Check function/method docstring."""
        # Ignore private/dunder methods unless they are __init__
        if node.name.startswith("_") and node.name != "__init__":
            self.generic_visit(node)
            return

        self._check_docstring(node, "Function")
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Check async function/method docstring."""
        if node.name.startswith("_") and node.name != "__init__":
            self.generic_visit(node)
            return

        self._check_docstring(node, "AsyncFunction")
        self.generic_visit(node)


def analyze_directory(target_dir: str) -> tuple[int, int, list[str]]:
    """
    Recursively analyze all Python files in the target directory.

    Returns:
        Tuple containing (total_nodes, documented_nodes, list_of_undocumented_items)
    """
    total_nodes = 0
    documented_nodes = 0
    all_undocumented: list[str] = []

    target_path = Path(target_dir)
    if not target_path.exists():
        print(f"Error: Directory '{target_dir}' does not exist.")
        sys.exit(1)

    python_files = list(target_path.rglob("*.py"))

    for py_file in python_files:
        # Skip test files and __init__.py if they are empty, but we check all src/ files
        try:
            with open(py_file, "r", encoding="utf-8") as f:
                source_code = f.read()

            tree = ast.parse(source_code, filename=str(py_file))
            visitor = DocstringVisitor(str(py_file))
            visitor.visit(tree)

            total_nodes += visitor.total_nodes
            documented_nodes += visitor.documented_nodes
            all_undocumented.extend(visitor.undocumented_items)

        except SyntaxError as e:
            print(f"Warning: Skipping {py_file} due to syntax error: {e}")
        except UnicodeDecodeError:
            print(f"Warning: Skipping {py_file} due to encoding issue.")

    return total_nodes, documented_nodes, all_undocumented


def main() -> None:
    """Main execution function for the docstring coverage checker."""
    print("=" * 60)
    print(" 📝 Docstring Coverage Checker")
    print("=" * 60)

    # Determine the src directory path relative to the script location
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent
    src_dir = project_root / "src"

    print(f"\n🔍 Analyzing directory: {src_dir}")

    total, documented, undocumented = analyze_directory(str(src_dir))

    if total == 0:
        print("\n⚠️  No Python files or AST nodes found to analyze.")
        sys.exit(1)

    coverage_percentage = (documented / total) * 100

    print("\n" + "-" * 60)
    print(" 📊 Coverage Report")
    print("-" * 60)
    print(f" Total AST Nodes Checked : {total}")
    print(f" Documented Nodes        : {documented}")
    print(f" Undocumented Nodes      : {total - documented}")
    print(f" Coverage Percentage     : {coverage_percentage:.2f}%")
    print("-" * 60)

    if coverage_percentage >= 80.0:
        print("\n✅ SUCCESS: Docstring coverage is >= 80%")

        if undocumented:
            print(
                "\nℹ️  Note: The following items are missing docstrings (for your reference):"
            )
            for item in undocumented[:10]:  # Show max 10 to avoid spam
                print(f"   - {item}")
            if len(undocumented) > 10:
                print(f"   ... and {len(undocumented) - 10} more.")

        sys.exit(0)
    else:
        print(
            f"\n❌ FAILURE: Docstring coverage is {coverage_percentage:.2f}% (Target: >= 80%)"
        )

        if undocumented:
            print("\n📋 List of undocumented items:")
            for item in undocumented:
                print(f"   - {item}")

        sys.exit(1)


if __name__ == "__main__":
    main()
