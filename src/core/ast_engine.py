"""
Source Code Abstract Syntax Tree (AST) Plagiarism Engine.

This module provides structural fingerprinting and similarity calculations for 
source code files (Python, C++, Java). It normalizes the code to ignore superficial 
differences (like variable renaming, strings, and comments) and generates 
k-gram based Winnowing fingerprints to detect structural similarities even if 
sections of the code are reordered or heavily obfuscated.
"""

import ast
import hashlib
import logging
import re
from dataclasses import dataclass
from typing import List, Set, Optional

logger = logging.getLogger(__name__)

# --- Configuration ---

@dataclass
class AstEngineConfig:
    """Configuration for the AST Fingerprinting Engine."""
    kgram_size: int = 15
    window_size: int = 4
    hash_limit: int = 8  # Substring length for the md5 hex digest
    
    # Structural keywords to retain for C++/Java parsing
    structural_keywords: frozenset = frozenset({
        'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'default',
        'break', 'continue', 'return', 'class', 'struct', 'enum',
        'public', 'private', 'protected', 'void', 'int', 'float',
        'double', 'char', 'boolean', 'bool', 'try', 'catch', 'finally',
        'throw', 'throws', 'new', 'static', 'final', 'const'
    })


# --- Python AST Normalization ---

class PythonAstNormalizer(ast.NodeVisitor):
    """
    Visits a Python AST and generates a normalized sequence of structural tokens.
    It replaces specific identifiers (like variable names and function names) 
    with generic placeholders so that renamed variables yield the same structure.
    """
    
    def __init__(self):
        self.structural_tokens: List[str] = []
        
    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.structural_tokens.append("FUNC_DEF")
        self.generic_visit(node)
        
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self.structural_tokens.append("ASYNC_FUNC_DEF")
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef):
        self.structural_tokens.append("CLASS_DEF")
        self.generic_visit(node)
        
    def visit_Return(self, node: ast.Return):
        self.structural_tokens.append("RETURN")
        self.generic_visit(node)
        
    def visit_Assign(self, node: ast.Assign):
        self.structural_tokens.append("ASSIGN")
        self.generic_visit(node)
        
    def visit_AugAssign(self, node: ast.AugAssign):
        self.structural_tokens.append("AUG_ASSIGN")
        self.generic_visit(node)
        
    def visit_For(self, node: ast.For):
        self.structural_tokens.append("FOR_LOOP")
        self.generic_visit(node)
        
    def visit_While(self, node: ast.While):
        self.structural_tokens.append("WHILE_LOOP")
        self.generic_visit(node)
        
    def visit_If(self, node: ast.If):
        self.structural_tokens.append("IF_STMT")
        self.generic_visit(node)
        
    def visit_With(self, node: ast.With):
        self.structural_tokens.append("WITH_CTX")
        self.generic_visit(node)
        
    def visit_Try(self, node: ast.Try):
        self.structural_tokens.append("TRY_CATCH")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name):
        # We ignore the actual variable name and just log that an identifier was used.
        self.structural_tokens.append("ID")
        self.generic_visit(node)
        
    def visit_arg(self, node: ast.arg):
        self.structural_tokens.append("ARG")
        self.generic_visit(node)
        
    def visit_Constant(self, node: ast.Constant):
        # Ignore literal values (strings, numbers, booleans)
        self.structural_tokens.append("CONST")
        self.generic_visit(node)

    def get_normalized_footprint(self) -> str:
        return " ".join(self.structural_tokens)


def parse_python_ast(source_code: str) -> str:
    """
    Parses Python source code into a normalized AST structural footprint.
    Handles malformed source code gracefully by returning an empty string.
    """
    try:
        tree = ast.parse(source_code)
        visitor = PythonAstNormalizer()
        visitor.visit(tree)
        return visitor.get_normalized_footprint()
    except Exception as e:
        logger.debug(f"Failed to parse Python AST: {e}")
        return ""


# --- C++ / Java Structural Lexing ---

def parse_cpp_java_structural(source_code: str, config: AstEngineConfig) -> str:
    """
    Lightweight structural extraction for C++ and Java.
    Acts as a pseudo-AST footprint by stripping out comments, literal strings,
    and specific variable names, while keeping control flow and structural tokens.
    """
    if not source_code:
        return ""
        
    # 1. Remove single-line comments (// ...)
    code = re.sub(r'//.*', '', source_code)
    
    # 2. Remove multi-line comments (/* ... */)
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    
    # 3. Remove string literals ("..." and '...')
    # This handles escaped quotes inside strings as well.
    code = re.sub(r'"(?:\\.|[^"\\])*"', ' STR_CONST ', code)
    code = re.sub(r"'(?:\\.|[^'\\])*'", ' CHAR_CONST ', code)
    
    # 4. Tokenize the remaining code.
    # We extract words (identifiers/keywords) and specific punctuation ({, }, (, ), ;, [, ]).
    tokens = re.findall(r'[a-zA-Z_]\w*|[{};()\[\]]', code)
    
    structural_footprint: List[str] = []
    
    for token in tokens:
        if token in config.structural_keywords:
            # Keep language keywords exactly as they are
            structural_footprint.append(token.upper())
        elif token in '{};()[]':
            # Keep structural punctuation
            structural_footprint.append(token)
        elif token in ('STR_CONST', 'CHAR_CONST'):
            structural_footprint.append(token)
        elif re.match(r'[a-zA-Z_]\w*', token):
            # Normalize user-defined identifiers (variables, custom classes, methods)
            structural_footprint.append('ID')
            
    return " ".join(structural_footprint)


def get_structural_footprint(source_code: str, filename: str, config: Optional[AstEngineConfig] = None) -> str:
    """
    Dispatches to the correct parsing engine based on the file extension.
    If the file is unsupported, it defaults to the C++/Java structural lexer 
    which serves as a robust fallback for most C-style languages.
    """
    if config is None:
        config = AstEngineConfig()
        
    filename_lower = filename.lower()
    
    if filename_lower.endswith('.py'):
        return parse_python_ast(source_code)
    elif filename_lower.endswith('.cpp') or filename_lower.endswith('.c') or \
         filename_lower.endswith('.h') or filename_lower.endswith('.hpp') or \
         filename_lower.endswith('.java'):
        return parse_cpp_java_structural(source_code, config)
    else:
        # Fallback for unsupported languages (e.g., JS, TS, C#) that share C-style syntax
        return parse_cpp_java_structural(source_code, config)


# --- Winnowing Fingerprinting ---

def generate_winnowing_fingerprint(text: str, config: Optional[AstEngineConfig] = None) -> Set[int]:
    """
    Generates a set of structural fingerprints using the Winnowing algorithm.
    
    Winnowing ensures that if two documents share a sufficiently long common 
    substring (k-gram), they are guaranteed to share at least one fingerprint.
    This is highly resilient to reordering of code blocks.
    
    Args:
        text: The normalized AST string/footprint.
        config: AST Engine Configuration.
        
    Returns:
        A set of integer hashes representing the document's structural fingerprint.
    """
    if config is None:
        config = AstEngineConfig()
        
    if not text:
        return set()
        
    # Remove all whitespace to create a dense structural sequence
    compact_text = re.sub(r'\s+', '', text)
    
    k = config.kgram_size
    w = config.window_size
    hash_limit = config.hash_limit
    
    # If the file is too small to form a single k-gram, hash the whole thing
    if len(compact_text) < k:
        return {int(hashlib.md5(compact_text.encode('utf-8')).hexdigest()[:hash_limit], 16)}
    
    # 1. Generate k-gram hashes
    kgram_hashes: List[int] = []
    for i in range(len(compact_text) - k + 1):
        kgram = compact_text[i : i + k]
        # Use MD5 and truncate to save memory while avoiding collisions
        h = int(hashlib.md5(kgram.encode('utf-8')).hexdigest()[:hash_limit], 16)
        kgram_hashes.append(h)
        
    # 2. Windowing (Winnowing proper)
    fingerprints: Set[int] = set()
    
    # Slide a window of size w over the k-gram hashes.
    # In each window, select the minimum hash value as a fingerprint.
    for i in range(len(kgram_hashes) - w + 1):
        window = kgram_hashes[i : i + w]
        fingerprints.add(min(window))
        
    return fingerprints


# --- Similarity Calculation ---

def calculate_ast_similarity(
    source_a: str, 
    filename_a: str, 
    source_b: str, 
    filename_b: str,
    config: Optional[AstEngineConfig] = None
) -> float:
    """
    Calculates the Jaccard similarity index between the AST fingerprints 
    of two source code files.
    
    Args:
        source_a: Raw source code of document A.
        filename_a: Filename of document A (used to determine language).
        source_b: Raw source code of document B.
        filename_b: Filename of document B.
        config: Optional configuration override.
        
    Returns:
        A similarity float between 0.0 (no overlap) and 1.0 (exact structural match).
    """
    if config is None:
        config = AstEngineConfig()
        
    fp_a = generate_winnowing_fingerprint(
        get_structural_footprint(source_a, filename_a, config), config
    )
    fp_b = generate_winnowing_fingerprint(
        get_structural_footprint(source_b, filename_b, config), config
    )
    
    if not fp_a and not fp_b:
        return 1.0  # Both empty -> exactly the same
    if not fp_a or not fp_b:
        return 0.0  # One empty, one not -> completely different
        
    intersection = fp_a.intersection(fp_b)
    union = fp_a.union(fp_b)
    
    if not union:
        return 0.0
        
    jaccard_similarity = len(intersection) / len(union)
    
    return float(round(jaccard_similarity, 4))


def calculate_containment_similarity(
    source_a: str, 
    filename_a: str, 
    source_b: str, 
    filename_b: str,
    config: Optional[AstEngineConfig] = None
) -> float:
    """
    Calculates the containment index. Useful for detecting if a smaller snippet 
    was entirely copied into a much larger file.
    
    Returns:
        The percentage of the smaller file's structural fingerprints that 
        are present in the larger file.
    """
    if config is None:
        config = AstEngineConfig()
        
    fp_a = generate_winnowing_fingerprint(
        get_structural_footprint(source_a, filename_a, config), config
    )
    fp_b = generate_winnowing_fingerprint(
        get_structural_footprint(source_b, filename_b, config), config
    )
    
    if not fp_a or not fp_b:
        return 0.0
        
    intersection = fp_a.intersection(fp_b)
    min_len = min(len(fp_a), len(fp_b))
    
    if min_len == 0:
        return 0.0
        
    containment = len(intersection) / min_len
    return float(round(containment, 4))
