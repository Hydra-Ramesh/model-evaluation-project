from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime, JSON, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
from .connection import Base

class Model(Base):
    __tablename__ = "models"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    provider = Column(String) # openai, anthropic, gemini, etc.
    parameters = Column(String, nullable=True) # e.g. "7B", "175B"

class Experiment(Base):
    __tablename__ = "experiments"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    config = Column(JSON) # e.g. {"temperature": [0.0, 0.5, 1.0]}

class BenchmarkRun(Base):
    __tablename__ = "benchmark_runs"
    id = Column(Integer, primary_key=True, index=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id"), nullable=True)
    model_id = Column(Integer, ForeignKey("models.id"))
    benchmark_name = Column(String, index=True) # e.g. "GSM8K", "MMLU"
    run_date = Column(DateTime, default=datetime.utcnow)
    configuration = Column(JSON) # e.g. {"temperature": 0.0, "max_tokens": 512}

    model = relationship("Model")
    experiment = relationship("Experiment")

class Question(Base):
    __tablename__ = "questions"
    id = Column(Integer, primary_key=True, index=True)
    benchmark_name = Column(String, index=True)
    category = Column(String, index=True)
    question_text = Column(String)
    expected_answer = Column(String)
    metadata_json = Column(JSON, nullable=True)

class Response(Base):
    __tablename__ = "responses"
    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(Integer, ForeignKey("benchmark_runs.id"))
    question_id = Column(Integer, ForeignKey("questions.id"))
    response_text = Column(String)
    latency_ms = Column(Float)
    prompt_tokens = Column(Integer)
    completion_tokens = Column(Integer)
    
    run = relationship("BenchmarkRun")
    question = relationship("Question")
    evaluation = relationship("EvaluationResult", back_populates="response", uselist=False)

class EvaluationResult(Base):
    __tablename__ = "evaluation_results"
    id = Column(Integer, primary_key=True, index=True)
    response_id = Column(Integer, ForeignKey("responses.id"))
    exact_match = Column(Boolean, nullable=True)
    semantic_score = Column(Float, nullable=True)
    bleu_score = Column(Float, nullable=True)
    rouge_score = Column(Float, nullable=True)
    bert_score = Column(Float, nullable=True)
    custom_metrics = Column(JSON, nullable=True)

    response = relationship("Response", back_populates="evaluation")

class HallucinationScore(Base):
    __tablename__ = "hallucination_scores"
    id = Column(Integer, primary_key=True, index=True)
    response_id = Column(Integer, ForeignKey("responses.id"))
    hallucination_probability = Column(Float)
    confidence = Column(Float)
    contradiction_detected = Column(Boolean)
    judge_model = Column(String)
    explanation = Column(String, nullable=True)

class CostTracking(Base):
    __tablename__ = "cost_tracking"
    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(Integer, ForeignKey("benchmark_runs.id"))
    total_cost = Column(Float)
    currency = Column(String, default="USD")
    calculation_method = Column(String)
