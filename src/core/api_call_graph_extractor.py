"""
src/core/api_call_graph_extractor.py
------------------------------------
API Call Graph and External Dependency Extractor.

Parses source code into directed graphs of external API calls and library
imports. This allows the system to detect algorithmic cloning based on
external resource utilization, bypassing variable renaming and control
flow obfuscation.
"""

import re
import ast
import logging
from typing import List, Dict, Any, Set, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class APICallNode:
    """Represents a node in the API call graph."""

    name: str
    module: str
    call_count: int = 1

    def get_id(self) -> str:
        return f"{self.module}.{self.name}"


@dataclass
class APICallGraph:
    """Represents the directed graph of API calls."""

    nodes: Dict[str, APICallNode] = field(default_factory=dict)
    edges: List[Tuple[str, str]] = field(default_factory=list)
    call_sequence: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": {
                k: {"module": v.module, "name": v.name, "count": v.call_count}
                for k, v in self.nodes.items()
            },
            "edges": self.edges,
            "call_sequence": self.call_sequence,
        }


def extract_python_api_graph(source_code: str) -> APICallGraph:
    """Extract API call graph from Python source code using the ast module.

    Identifies import statements and function calls to external modules.

    Args:
        source_code: The Python source code string.

    Returns:
        An APICallGraph object representing the dependency chain.
    """
    graph = APICallGraph()

    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        logger.warning("Failed to parse Python code for API graph: %s", e)
        return graph

    # Track imported modules and their aliases
    imports = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                imports[name] = alias.name

        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                imports[name] = f"{module}.{name}"

    # Extract function calls
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            module_name = ""
            func_name = ""

            if isinstance(node.func, ast.Name):
                # Direct call: func()
                func_name = node.func.id
                if func_name in imports:
                    module_name = imports[func_name]
                else:
                    module_name = "builtins"

            elif isinstance(node.func, ast.Attribute):
                # Attribute call: module.func() or obj.method()
                if isinstance(node.func.value, ast.Name):
                    base_name = node.func.value.id
                    func_name = node.func.attr
                    if base_name in imports:
                        module_name = imports[base_name]
                    else:
                        module_name = base_name
                else:
                    # Complex chained calls, just get the final attribute
                    func_name = node.func.attr
                    module_name = "unknown"

            if func_name:
                node_id = f"{module_name}.{func_name}"
                if node_id not in graph.nodes:
                    graph.nodes[node_id] = APICallNode(
                        name=func_name, module=module_name
                    )
                else:
                    graph.nodes[node_id].call_count += 1

                graph.call_sequence.append(node_id)

    # Build edges based on sequential calls
    for i in range(len(graph.call_sequence) - 1):
        edge = (graph.call_sequence[i], graph.call_sequence[i + 1])
        if edge not in graph.edges:
            graph.edges.append(edge)

    logger.info(
        "Extracted API graph with %d nodes and %d edges.",
        len(graph.nodes),
        len(graph.edges),
    )
    return graph
