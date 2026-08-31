import pytest
from src.core.ast_engine import parse_python_ast, parse_cpp_java_ast, winnowing_fingerprint, calculate_ast_similarity

def test_parse_python_ast():
    code = "def test(a):\n    b = a + 1\n    return b"
    ast_str = parse_python_ast(code)
    assert "FUNC" in ast_str
    assert "VAR" in ast_str

def test_parse_cpp_java_ast():
    code = "public class Test { public static void main(String[] args) { int a = 1; } }"
    ast_str = parse_cpp_java_ast(code)
    assert "public" in ast_str
    assert "class" in ast_str
    assert "int" in ast_str
    assert "{" in ast_str
    assert "}" in ast_str

def test_winnowing_fingerprint():
    code = "public class Test { }"
    fp = winnowing_fingerprint(code, k=5, w=2)
    assert len(fp) > 0

def test_calculate_ast_similarity():
    code_a = "def func1(x):\n    y = x * 2\n    return y"
    code_b = "def my_func(a):\n    b = a * 2\n    return b"
    
    # Should be highly similar since variables are normalized
    sim = calculate_ast_similarity(code_a, "test1.py", code_b, "test2.py")
    assert sim > 0.8
