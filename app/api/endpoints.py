from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from ..database.connection import get_db
from ..database.models import Model, Experiment, BenchmarkRun, EvaluationResult
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

@router.get("/leaderboard")
def get_leaderboard(db: Session = Depends(get_db)):
    # Future integration: Elo rating calculation across models
    return {"message": "Leaderboard data endpoint"}

@router.get("/hallucinations")
def get_hallucinations(db: Session = Depends(get_db)):
    return {"message": "Hallucination statistics endpoint"}

@router.get("/experiments")
def list_experiments(db: Session = Depends(get_db)):
    return db.query(Experiment).all()
