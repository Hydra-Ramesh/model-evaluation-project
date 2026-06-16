import os
import pandas as pd
from sqlalchemy.orm import Session
from ..database.connection import SessionLocal
from ..database.models import EvaluationResult, BenchmarkRun, Model, Response, Question

def fetch_training_data() -> pd.DataFrame:
    db: Session = SessionLocal()
    
    query = db.query(
        Question.question_text.label("question"),
        Question.expected_answer.label("ground_truth"),
        Response.response_text.label("model_response"),
        Response.prompt_tokens,
        Response.completion_tokens,
        Response.latency_ms,
        EvaluationResult.exact_match,
        EvaluationResult.semantic_score,
        EvaluationResult.custom_metrics.label("metadata_json"),
        Model.name.label("model_name"),
        BenchmarkRun.benchmark_name
    ).select_from(EvaluationResult)\
     .join(Response, EvaluationResult.response_id == Response.id)\
     .join(Question, Response.question_id == Question.id)\
     .join(BenchmarkRun, Response.run_id == BenchmarkRun.id)\
     .join(Model, BenchmarkRun.model_id == Model.id)
     
    records = query.all()
    db.close()
    
    if not records:
        raise ValueError("No evaluation records found.")
        
    df = pd.DataFrame(records)
    
    df['temperature'] = df['metadata_json'].apply(lambda x: x.get('temperature', 0.0) if x else 0.0)
    df['max_tokens'] = df['metadata_json'].apply(lambda x: x.get('max_tokens', 1024) if x else 1024)
    df['quality_score'] = df['metadata_json'].apply(lambda x: x.get('quality_score', 5.0) if x else 5.0)
    df['cost'] = df['metadata_json'].apply(lambda x: x.get('cost', 0.001) if x else 0.001)
    df['hallucination_detected'] = df['metadata_json'].apply(lambda x: x.get('hallucination_detected', False) if x else False)
    df['hallucination_detected'] = df['hallucination_detected'].astype(int)
    
    df = df.drop(columns=['metadata_json'])
    
    return df
