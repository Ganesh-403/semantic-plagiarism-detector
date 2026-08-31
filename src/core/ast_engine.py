import ast
import hashlib
import re

def parse_python_ast(source_code: str) -> str:
    """Parses Python source code into a normalized AST string."""
    try:
        tree = ast.parse(source_code)
        # Normalize: replace variable names, remove docstrings
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                node.id = "VAR"
            elif isinstance(node, ast.arg):
                node.arg = "ARG"
            elif isinstance(node, ast.FunctionDef):
                node.name = "FUNC"
        return ast.dump(tree)
    except Exception:
        return ""

def parse_cpp_java_ast(source_code: str) -> str:
    """
    Lightweight structural extraction for C++/Java that acts as an AST footprint.
    Ignores comments, strings, and variable names, retaining structural tokens.
    """
    # Remove single line and multi-line comments
    code = re.sub(r'//.*', '', source_code)
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    # Remove strings
    code = re.sub(r'".*?"', 'STR', code)
    code = re.sub(r"'.*?'", 'STR', code)
    
    # Extract structural keywords and braces
    keywords = {
        'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'return', 
        'class', 'public', 'private', 'protected', 'void', 'int', 'float', 
        'double', 'char', 'boolean'
    }
    tokens = re.findall(r'[a-zA-Z_]\w*|[{};()]', code)
    
    ast_footprint = []
    for token in tokens:
        if token in keywords or token in '{};()':
            ast_footprint.append(token)
        elif re.match(r'[a-zA-Z_]\w*', token):
            ast_footprint.append('ID')
            
    return ' '.join(ast_footprint)

def get_ast_footprint(source_code: str, filename: str) -> str:
    """Gets the normalized AST footprint based on file extension."""
    filename_lower = filename.lower()
    if filename_lower.endswith('.py'):
        return parse_python_ast(source_code)
    elif filename_lower.endswith('.cpp') or filename_lower.endswith('.java'):
        return parse_cpp_java_ast(source_code)
    else:
        # Gracefully handle unsupported source files by treating them as structural tokens
        return parse_cpp_java_ast(source_code)

def winnowing_fingerprint(text: str, k: int = 15, w: int = 4) -> set:
    """Winnowing algorithm for k-gram fingerprinting."""
    if not text:
        return set()
    # clean text
    text = re.sub(r'\s+', '', text)
    if len(text) < k:
        return {hashlib.md5(text.encode()).hexdigest()}
    
    # Generate k-grams hashes
    hashes = []
    for i in range(len(text) - k + 1):
        kgram = text[i:i+k]
        h = int(hashlib.md5(kgram.encode()).hexdigest()[:8], 16)
        hashes.append(h)
        
    # Windows
    fingerprints = set()
    for i in range(len(hashes) - w + 1):
        window = hashes[i:i+w]
        fingerprints.add(min(window))
        
    return fingerprints

def calculate_ast_similarity(source_a: str, filename_a: str, source_b: str, filename_b: str) -> float:
    """Calculates AST similarity between two source code files."""
    fp1 = winnowing_fingerprint(get_ast_footprint(source_a, filename_a))
    fp2 = winnowing_fingerprint(get_ast_footprint(source_b, filename_b))
    
    if not fp1 or not fp2:
        return 0.0
        
    intersection = fp1.intersection(fp2)
    union = fp1.union(fp2)
    return len(intersection) / len(union) if union else 0.0
