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

"""Neural Code Clone Domain Model.

Defines data classes for source code AST embedding representations, token-level
similarity scores, obfuscation detection telemetry, and code clone match records.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class CodeAstEmbedding:
    """Represents abstract syntax tree embeddings for code similarity matching."""

    file_id: str
    source_language: str  # e.g., 'python', 'javascript', 'java', 'cpp'
    ast_token_count: int
    cyclomatic_complexity: int
    vector_embedding: List[float] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class CodeCloneMatch:
    """Represents a detected code clone match between two source files."""

    clone_id: str
    source_file_id: str
    target_file_id: str
    ast_similarity_score: float  # Range: 0.0 - 1.0
    token_overlap_score: float  # Range: 0.0 - 1.0
    neural_semantic_similarity: float  # Range: 0.0 - 1.0
    overall_clone_score: float  # Weighted ensemble score
    clone_type: str  # 'Type-1 (Exact)', 'Type-2 (Renamed)', 'Type-3 (Modified AST)', 'Type-4 (Semantic Equivalent)'
    obfuscation_detected: bool = False
    detected_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class CodeCloneScanReport:
    """Audit summary report for code clone analysis."""

    report_id: str
    repository_name: str
    total_files_scanned: int
    total_clones_found: int
    highest_similarity_ratio: float
    scanned_at: datetime = field(default_factory=datetime.utcnow)
    clone_matches: List[CodeCloneMatch] = field(default_factory=list)
