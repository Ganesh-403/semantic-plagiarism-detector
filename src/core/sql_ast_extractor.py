"""
src/core/sql_ast_extractor.py
-----------------------------
SQL AST and Schema Dependency Extractor.

Parses SQL queries into normalized Abstract Syntax Trees (ASTs) and extracts
table/column dependency graphs. Normalizes identifiers (tables, columns, aliases)
to detect cloned database logic even when names are obfuscated.
"""

import re
import logging
from typing import List, Dict, Any, Set, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SQLNode:
    """Represents a clause or operation in the SQL AST."""

    clause_type: str  # SELECT, FROM, JOIN, WHERE, GROUP_BY, ORDER_BY
    normalized_tokens: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"clause_type": self.clause_type, "tokens": self.normalized_tokens}


@dataclass
class SQLAST:
    """Represents the normalized AST and schema dependencies of a SQL query."""

    nodes: List[SQLNode] = field(default_factory=list)
    tables: Set[str] = field(default_factory=set)
    columns: Set[str] = field(default_factory=set)
    aliases: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "tables": list(self.tables),
            "columns": list(self.columns),
        }


# Regex patterns for SQL clause extraction
CLAUSE_PATTERNS = {
    "SELECT": re.compile(r"\bSELECT\s+(.*?)\bFROM\b", re.IGNORECASE | re.DOTALL),
    "FROM": re.compile(
        r"\bFROM\s+([a-zA-Z0-9_,\s]+?)(?:\bWHERE\b|\bGROUP\b|\bORDER\b|\bLIMIT\b|\bJOIN\b|$)",
        re.IGNORECASE | re.DOTALL,
    ),
    "JOIN": re.compile(
        r"\bJOIN\s+([a-zA-Z0-9_]+)\s+(?:AS\s+)?([a-zA-Z0-9_]*)\s+ON\b", re.IGNORECASE
    ),
    "WHERE": re.compile(
        r"\bWHERE\s+(.*?)(?:\bGROUP\b|\bORDER\b|\bLIMIT\b|$)", re.IGNORECASE | re.DOTALL
    ),
    "GROUP_BY": re.compile(
        r"\bGROUP\s+BY\s+(.*?)(?:\bHAVING\b|\bORDER\b|\bLIMIT\b|$)",
        re.IGNORECASE | re.DOTALL,
    ),
    "ORDER_BY": re.compile(
        r"\bORDER\s+BY\s+(.*?)(?:\bLIMIT\b|$)", re.IGNORECASE | re.DOTALL
    ),
}


def _normalize_identifier(name: str, id_map: Dict[str, str], prefix: str) -> str:
    """Normalize a SQL identifier (table/column) to an abstract placeholder."""
    name = name.strip().lower()
    if not name or name in ("*", "null", "true", "false"):
        return name
    if name not in id_map:
        id_map[name] = f"{prefix}_{len(id_map) + 1}"
    return id_map[name]


def extract_sql_ast(sql_query: str) -> SQLAST:
    """Parse a SQL query and extract a normalized AST and schema dependencies.

    Args:
        sql_query: Raw SQL query string.

    Returns:
        A SQLAST object containing normalized nodes and dependencies.
    """
    if not sql_query or not isinstance(sql_query, str):
        return SQLAST()

    ast = SQLAST()
    table_map = {}
    col_map = {}

    # Extract tables from FROM and JOIN clauses
    from_match = CLAUSE_PATTERNS["FROM"].search(sql_query)
    if from_match:
        raw_tables = re.findall(r"\b([a-zA-Z0-9_]+)\b", from_match.group(1))
        for tbl in raw_tables:
            norm_tbl = _normalize_identifier(tbl, table_map, "TBL")
            ast.tables.add(norm_tbl)

    join_matches = CLAUSE_PATTERNS["JOIN"].finditer(sql_query)
    for match in join_matches:
        norm_tbl = _normalize_identifier(match.group(1), table_map, "TBL")
        ast.tables.add(norm_tbl)
        if match.group(2):
            ast.aliases[match.group(2).lower()] = norm_tbl

    # Extract and normalize clauses
    for clause_type, pattern in CLAUSE_PATTERNS.items():
        matches = pattern.finditer(sql_query)
        for match in matches:
            raw_text = match.group(1) if match.lastindex else match.group(0)

            # Tokenize and normalize identifiers in the clause
            tokens = re.findall(r"\b([a-zA-Z0-9_]+)\b", raw_text)
            norm_tokens = []

            for token in tokens:
                token_lower = token.lower()
                # Skip SQL keywords
                if token_lower in (
                    "as",
                    "on",
                    "and",
                    "or",
                    "in",
                    "not",
                    "like",
                    "between",
                    "asc",
                    "desc",
                ):
                    norm_tokens.append(token_lower)
                    continue

                # Check if it's a known table or alias
                if token_lower in table_map or token_lower in ast.aliases:
                    norm_tokens.append(
                        table_map.get(token_lower, ast.aliases[token_lower])
                    )
                else:
                    # Assume it's a column
                    norm_col = _normalize_identifier(token, col_map, "COL")
                    ast.columns.add(norm_col)
                    norm_tokens.append(norm_col)

            if norm_tokens:
                ast.nodes.append(
                    SQLNode(clause_type=clause_type, normalized_tokens=norm_tokens)
                )

    # Sort nodes by a standard execution order for consistent comparison
    execution_order = {
        "SELECT": 1,
        "FROM": 2,
        "JOIN": 3,
        "WHERE": 4,
        "GROUP_BY": 5,
        "ORDER_BY": 6,
    }
    ast.nodes.sort(key=lambda n: execution_order.get(n.clause_type, 99))

    logger.info(
        "Extracted SQL AST with %d nodes, %d tables, %d columns.",
        len(ast.nodes),
        len(ast.tables),
        len(ast.columns),
    )
    return ast
