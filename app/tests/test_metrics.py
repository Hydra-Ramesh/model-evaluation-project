import pytest
from app.evaluation.metrics import EvaluationEngine

@pytest.fixture
def engine():
    # Use the lightweight default model for testing
    return EvaluationEngine()

def test_exact_match(engine):
    assert engine.exact_match("42", "42") is True
    assert engine.exact_match(" 42 ", "42") is True
    assert engine.exact_match("Forty Two", "42") is False

def test_semantic_similarity(engine):
    # These sentences are similar
    sim1 = engine.semantic_similarity("The cat sat on the mat.", "A cat is sitting on a mat.")
    # These sentences are entirely different
    sim2 = engine.semantic_similarity("The cat sat on the mat.", "The rocket launched into space.")
    
    assert sim1 > sim2
    assert sim1 > 0.7  # High similarity expected
    assert sim2 < 0.5  # Low similarity expected
