"""
src/core/notebook_graph_extractor.py
------------------------------------
Jupyter Notebook Execution Graph and Data Lineage Extractor.

Parses `.ipynb` JSON structures to extract cell execution sequences and
maps variable data lineage (definitions and usages) to build a Directed
Acyclic Graph (DAG) representing the computational workflow.
"""

import json
import re
import logging
from typing import List, Dict, Any, Set, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class NotebookCell:
    """Represents a single cell in a Jupyter Notebook."""

    cell_id: str
    cell_type: str  # 'code', 'markdown', 'raw'
    source: str
    execution_count: int | None
    defines: Set[str] = field(default_factory=set)
    uses: Set[str] = field(default_factory=set)


@dataclass
class NotebookGraph:
    """Represents the execution and data lineage graph of a notebook."""

    cells: List[NotebookCell] = field(default_factory=list)
    execution_sequence: List[int] = field(default_factory=list)
    lineage_edges: List[Tuple[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cell_count": len(self.cells),
            "execution_sequence": self.execution_sequence,
            "lineage_edges": self.lineage_edges,
        }


# Regex patterns for simple Python variable assignment and usage extraction
# This is a lightweight heuristic proxy for a full AST parser
ASSIGNMENT_PATTERN = re.compile(r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=(?!=)", re.MULTILINE)
USAGE_PATTERN = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b")

# Python built-ins and keywords to ignore during lineage mapping
PYTHON_BUILTINS = {
    "print",
    "len",
    "range",
    "list",
    "dict",
    "set",
    "tuple",
    "int",
    "float",
    "str",
    "bool",
    "type",
    "isinstance",
    "def",
    "class",
    "return",
    "if",
    "else",
    "elif",
    "for",
    "while",
    "in",
    "not",
    "and",
    "or",
    "True",
    "False",
    "None",
    "import",
    "from",
    "as",
    "with",
    "try",
    "except",
    "finally",
    "raise",
    "yield",
}


def _extract_variables(source: str) -> Tuple[Set[str], Set[str]]:
    """Extract defined and used variables from Python source code."""
    defines = set()
    for match in ASSIGNMENT_PATTERN.finditer(source):
        var_name = match.group(1)
        if var_name not in PYTHON_BUILTINS:
            defines.add(var_name)

    uses = set()
    for match in USAGE_PATTERN.finditer(source):
        var_name = match.group(1)
        if var_name not in PYTHON_BUILTINS and var_name not in defines:
            uses.add(var_name)

    return defines, uses


def extract_notebook_graph(notebook_json: str) -> NotebookGraph:
    """Parse a Jupyter Notebook JSON and extract the execution graph.

    Args:
        notebook_json: Raw JSON string of the .ipynb file.

    Returns:
        A NotebookGraph object containing cells, execution sequence, and lineage edges.
    """
    try:
        data = json.loads(notebook_json)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse notebook JSON: %s", e)
        return NotebookGraph()

    cells = []
    execution_sequence = []

    # Extract cells and their variables
    for i, cell_data in enumerate(data.get("cells", [])):
        cell_type = cell_data.get("cell_type", "raw")
        source_list = cell_data.get("source", [])
        source = (
            "".join(source_list) if isinstance(source_list, list) else str(source_list)
        )

        exec_count = cell_data.get("execution_count")
        if cell_type == "code" and isinstance(exec_count, int):
            execution_sequence.append(exec_count)

        defines, uses = set(), set()
        if cell_type == "code":
            defines, uses = _extract_variables(source)

        cell_id = cell_data.get("id", f"cell_{i}")
        cells.append(
            NotebookCell(
                cell_id=str(cell_id),
                cell_type=cell_type,
                source=source,
                execution_count=exec_count,
                defines=defines,
                uses=uses,
            )
        )

    # Map data lineage edges (which cell defines a variable used by another)
    lineage_edges = []
    var_providers = {}  # Maps variable name to the cell_id that last defined it

    # We sort cells by execution count to establish temporal lineage
    code_cells = [
        c for c in cells if c.cell_type == "code" and c.execution_count is not None
    ]
    code_cells.sort(key=lambda c: c.execution_count)

    for cell in code_cells:
        for var in cell.uses:
            if var in var_providers:
                lineage_edges.append((var_providers[var], cell.cell_id))
        for var in cell.defines:
            var_providers[var] = cell.cell_id

    logger.info(
        "Extracted notebook graph with %d cells and %d lineage edges.",
        len(cells),
        len(lineage_edges),
    )

    return NotebookGraph(
        cells=cells, execution_sequence=execution_sequence, lineage_edges=lineage_edges
    )
