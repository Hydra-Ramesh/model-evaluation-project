from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from ..database.connection import get_db
from ..database.models import Model, Experiment, BenchmarkRun, EvaluationResult, Question
from .schemas import RunBenchmarkRequest, RunBenchmarkResponse, ExperimentCreate, ExperimentResponse
from ..services.runner import ExperimentRunner
from ..evaluation.metrics import EvaluationEngine
from ..models.universal_connector import UniversalModelConnector 

router = APIRouter()

# Instantiate the global evaluation engine
eval_engine = EvaluationEngine()

def get_runner(model_name: str, db: Session):
    # Dependency injection for the orchestrator
    # The connector is now instantiated dynamically based on the requested model name
    connector = UniversalModelConnector(model_name=model_name) 
    return ExperimentRunner(db, connector, eval_engine)

@router.post("/run-benchmark", response_model=RunBenchmarkResponse)
def run_benchmark(req: RunBenchmarkRequest, db: Session = Depends(get_db)):
    model = db.query(Model).filter(Model.id == req.model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
        
    runner = get_runner(model.name, db)
    run = runner.run_benchmark(req.model_id, req.benchmark_name, req.config)
    return RunBenchmarkResponse(run_id=run.id, status="started")

@router.post("/experiment", response_model=ExperimentResponse)
def create_experiment(req: ExperimentCreate, db: Session = Depends(get_db)):
    exp = Experiment(name=req.name, description=req.description, config=req.config)
    db.add(exp)
    db.commit()
    db.refresh(exp)
    
    # Trigger execution
    runner.run_experiment(exp.id, req.model_ids, req.benchmark_names)
    
    return ExperimentResponse(experiment_id=exp.id, message="Experiment triggered successfully")

@router.get("/models")
def list_models(db: Session = Depends(get_db)):
    return db.query(Model).all()

@router.get("/results")
def get_results(run_id: int, db: Session = Depends(get_db)):
    # Simple join to get evaluations for a run (requires Response join, omitted for brevity)
    return {"message": f"Results for run {run_id}"}

@router.get("/overview")
def get_overview(db: Session = Depends(get_db)):
    c1 = db.query(Model).count()
    c2 = db.query(BenchmarkRun).count()
    c3 = db.query(Question).count()
    return {"models": c1, "runs": c2, "questions": c3}

@router.get("/leaderboard")
def get_leaderboard(db: Session = Depends(get_db)):
    from sqlalchemy import text
    query = text("""
        SELECT m.name as model, 
               SUM(CASE WHEN e.exact_match THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as accuracy
        FROM evaluation_results e
        JOIN responses r ON e.response_id = r.id
        JOIN benchmark_runs b ON r.run_id = b.id
        JOIN models m ON b.model_id = m.id
        GROUP BY m.name
        ORDER BY accuracy DESC
    """)
    try:
        result = db.execute(query).fetchall()
        leaderboard = []
        for row in result:
            accuracy = float(row.accuracy) if row.accuracy is not None else 50.0
            elo = 1500 + (accuracy - 50.0) * 10
            leaderboard.append({
                "Model": row.model,
                "Avg Accuracy (%)": accuracy,
                "Elo Rating": elo
            })
        return leaderboard
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@router.get("/cost-performance")
def get_cost_performance(db: Session = Depends(get_db)):
    from sqlalchemy import text
    cost_col = "json_extract(e.custom_metrics, '$.cost')" if db.bind.dialect.name == "sqlite" else "(e.custom_metrics->>'cost')"
    query = text(f"""
        SELECT m.name as model, 
               AVG(CAST({cost_col} AS FLOAT)) * 1000 as cost_1k,
               SUM(CASE WHEN e.exact_match THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as accuracy
        FROM evaluation_results e
        JOIN responses r ON e.response_id = r.id
        JOIN benchmark_runs b ON r.run_id = b.id
        JOIN models m ON b.model_id = m.id
        GROUP BY m.name
    """)
    try:
        result = db.execute(query).fetchall()
        cost_data = []
        for row in result:
            cost_data.append({
                "Model": row.model,
                "Cost per 1k Tokens ($)": float(row.cost_1k) if row.cost_1k is not None else 0.0,
                "Avg Accuracy (%)": float(row.accuracy) if row.accuracy is not None else 50.0
            })
        return cost_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@router.get("/hallucinations")
def get_hallucinations(db: Session = Depends(get_db)):
    from sqlalchemy import text
    if db.bind.dialect.name == "sqlite":
        is_hal_expr = "CAST(json_extract(e.custom_metrics, '$.hallucination_detected') AS FLOAT)"
        hal_filter = "json_extract(e.custom_metrics, '$.hallucination_detected')"
    else:
        is_hal_expr = "CASE WHEN (e.custom_metrics->>'hallucination_detected') = 'true' THEN 1.0 ELSE 0.0 END"
        hal_filter = "(e.custom_metrics->>'hallucination_detected')"
        
    query = text(f"""
        SELECT m.name as model, 
               {is_hal_expr} as is_hal
        FROM evaluation_results e
        JOIN responses r ON e.response_id = r.id
        JOIN benchmark_runs b ON r.run_id = b.id
        JOIN models m ON b.model_id = m.id
        WHERE {hal_filter} IS NOT NULL
    """)
    try:
        result = db.execute(query).fetchall()
        hallucinations = []
        for row in result:
            hallucinations.append({
                "Model": row.model,
                "is_hal": float(row.is_hal) if row.is_hal is not None else 0.0
            })
        return hallucinations
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@router.get("/model-stats/{model_name}")
def get_model_stats(model_name: str, db: Session = Depends(get_db)):
    from sqlalchemy import text
    cost_col = "json_extract(e.custom_metrics, '$.cost')" if db.bind.dialect.name == "sqlite" else "(e.custom_metrics->>'cost')"
    
    if db.bind.dialect.name == "sqlite":
        hal_expr = "CAST(json_extract(e.custom_metrics, '$.hallucination_detected') AS FLOAT)"
    else:
        hal_expr = "CASE WHEN (e.custom_metrics->>'hallucination_detected') = 'true' THEN 1.0 ELSE 0.0 END"
    
    query = text(f"""
        SELECT 
            AVG(r.latency_ms) / 1000.0 as avg_lat,
            SUM(CASE WHEN e.exact_match THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as acc,
            AVG(CAST({cost_col} AS FLOAT)) * 1000 as cost_1k,
            AVG({hal_expr}) * 100.0 as hal_rate
        FROM evaluation_results e
        JOIN responses r ON e.response_id = r.id
        JOIN benchmark_runs b ON r.run_id = b.id
        JOIN models m ON b.model_id = m.id
        WHERE m.name = :model_name
    """)
    try:
        row = db.execute(query, {"model_name": model_name}).fetchone()
        if not row or row.acc is None:
            return {
                "avg_lat": 0.0,
                "acc": 0.0,
                "cost_1k": 0.0,
                "hal_rate": 0.0
            }
        return {
            "avg_lat": float(row.avg_lat) if row.avg_lat is not None else 0.0,
            "acc": float(row.acc) if row.acc is not None else 0.0,
            "cost_1k": float(row.cost_1k) if row.cost_1k is not None else 0.0,
            "hal_rate": float(row.hal_rate) if row.hal_rate is not None else 0.0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@router.get("/experiments")
def list_experiments(db: Session = Depends(get_db)):
    return db.query(Experiment).all()
