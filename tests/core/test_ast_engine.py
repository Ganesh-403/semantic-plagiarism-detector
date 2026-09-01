import pytest
from src.core.ast_engine import (
    AstEngineConfig,
    parse_python_ast,
    parse_cpp_java_structural,
    get_structural_footprint,
    generate_winnowing_fingerprint,
    calculate_ast_similarity,
    calculate_containment_similarity
)

class TestPythonAstNormalizer:
    def test_basic_normalization(self):
        code_1 = "def my_func(a, b):\n    c = a + b\n    return c"
        code_2 = "def addition(x, y):\n    z = x + y\n    return z"
        
        ast_1 = parse_python_ast(code_1)
        ast_2 = parse_python_ast(code_2)
        
        # Structure should be exactly the same despite different variable names
        assert ast_1 == ast_2
        assert "FUNC_DEF" in ast_1
        assert "RETURN" in ast_1

    def test_syntax_error_handling(self):
        code = "def malformed(x"
        ast_str = parse_python_ast(code)
        assert ast_str == ""  # Graceful fallback


class TestCppJavaStructuralLexer:
    def test_comment_and_string_removal(self):
        config = AstEngineConfig()
        code = '''
        // Single line comment
        public class Main {
            /* Multi 
               line 
               comment */
            public static void main(String[] args) {
                String s = "Hello, World!";
                int x = 5;
            }
        }
        '''
        footprint = parse_cpp_java_structural(code, config)
        
        assert "PUBLIC" in footprint
        assert "CLASS" in footprint
        assert "VOID" in footprint
        assert "INT" in footprint
        assert "STR_CONST" in footprint
        assert "Hello" not in footprint
        assert "Single line comment" not in footprint

    def test_structural_equivalence(self):
        config = AstEngineConfig()
        code_1 = "int add(int a, int b) { return a + b; }"
        code_2 = "int sum(int x, int y) { return x + y; }"
        
        fp_1 = parse_cpp_java_structural(code_1, config)
        fp_2 = parse_cpp_java_structural(code_2, config)
        
        # Identifiers are normalized to 'ID'
        assert fp_1 == fp_2


class TestFingerprintingAndSimilarity:
    def test_winnowing_fingerprint_small_file(self):
        code = "INT ID ( ) { RETURN CONST ; }"
        fp = generate_winnowing_fingerprint(code, AstEngineConfig(kgram_size=50))
        assert len(fp) == 1  # Less than kgram size hashes the whole thing

    def test_winnowing_fingerprint_large_file(self):
        code = "ID " * 100
        fp = generate_winnowing_fingerprint(code, AstEngineConfig(kgram_size=10, window_size=4))
        assert len(fp) > 0

    def test_calculate_ast_similarity(self):
        code_1 = "def func(x): return x * 2"
        code_2 = "def func(y): return y * 2"
        
        sim = calculate_ast_similarity(code_1, "test.py", code_2, "test.py")
        assert sim == 1.0  # Perfect structural match

    def test_calculate_containment_similarity(self):
        code_small = "def snippet(x): return x"
        code_large = "def other(): pass\n\ndef snippet(y): return y\n\ndef extra(): pass"
        
        sim = calculate_containment_similarity(code_small, "small.py", code_large, "large.py")
        # The small snippet is fully contained within the large file's structure
        assert sim == 1.0

    def test_unsupported_language_fallback(self):
        # Even unsupported languages should fallback to the C-style lexer gracefully
        code = "function jsTest() { console.log('hello'); }"
        fp = get_structural_footprint(code, "test.js")
        assert "ID" in fp
        assert "(" in fp
        assert ")" in fp
        assert "{" in fp
        assert "}" in fp
        assert "STR_CONST" in fp
