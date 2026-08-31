import pytest
from src.analysis.ast_hashing_engine import ASTHashingEngine
from src.analysis.clone_orchestrator import CloneOrchestrator

def test_ast_hash_ignores_variable_name_changes():
    engine = ASTHashingEngine()
    
    code_alpha = """
def compute_sum(alpha, beta):
    result_data = alpha + beta
    return result_data
"""
    code_beta = """
def compute_sum(x, y):
    total_val = x + y
    return total_val
"""
    
    res_a = engine.generate_fingerprint(code_alpha)
    res_b = engine.generate_fingerprint(code_beta)
    
    assert res_a["success"] is True
    assert res_b["success"] is True
    # Acceptance Criteria: Verify that variable rewriting results in identical hashes
    assert res_a["ast_hash"] == res_b["ast_hash"]

def test_syntax_errors_handled_gracefully():
    engine = ASTHashingEngine()
    corrupted_code = "def unclosed_function_block(x:"
    
    res = engine.generate_fingerprint(corrupted_code)
    
    assert res["success"] is False
    assert "Syntax parsing exception" in res["error"]

def test_clone_orchestrator_classification():
    orchestrator = CloneOrchestrator()
    
    base_code = "def calculate(a):\n    return a * 2"
    historical = [{
        "submission_id": "hist_01",
        "source_code": "def calculate(a):\n    return a * 2",
        "tokens": ["FunctionDefStart", "Arg:var_1", "Name:var_1", "FunctionDefEnd"]
    }]
    
    matches = orchestrator.process_and_classify_submission(base_code, historical)
    assert len(matches) == 1
    assert matches[0]["classification"] == "type_1_exact"
