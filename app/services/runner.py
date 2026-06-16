import logging
from typing import Dict, Any, List
from sqlalchemy.orm import Session
from ..database.models import Experiment, BenchmarkRun
from ..models.base import BaseModelConnector
from ..evaluation.metrics import EvaluationEngine

logger = logging.getLogger(__name__)

class ExperimentRunner:
    """
    Orchestrates the execution of benchmarks and experiments.
    In a production system, this would queue tasks to Celery or RQ.
    """
    def __init__(self, db: Session, model_connector: BaseModelConnector, eval_engine: EvaluationEngine):
        self.db = db
        self.model_connector = model_connector
        self.eval_engine = eval_engine

    def run_benchmark(self, model_id: int, benchmark_name: str, config: Dict[str, Any], experiment_id: int = None) -> BenchmarkRun:
        """
        Runs a specific benchmark configuration for a single model.
        """
        # Create the run record
        run = BenchmarkRun(
            model_id=model_id,
            benchmark_name=benchmark_name,
            configuration=config,
            experiment_id=experiment_id
        )
        self.db.add(run)
        self.db.commit()
        self.db.refresh(run)

        # Simulation of pipeline execution:
        # 1. Fetch questions using specific BenchmarkLoader
        # 2. Iterate and generate answers via self.model_connector.generate()
        # 3. Save Responses and usage stats to DB
        # 4. Run evaluations using self.eval_engine.evaluate_all()
        # 5. Save EvaluationResults to DB
        
        logger.info(f"Started benchmark run {run.id} for model {model_id} on {benchmark_name}")
        return run

    def run_experiment(self, experiment_id: int, model_ids: List[int], benchmark_names: List[str]):
        """
        Executes a full parameterized experiment across multiple models and datasets.
        """
        experiment = self.db.query(Experiment).filter(Experiment.id == experiment_id).first()
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")

        # Extract parameters for sweeps (e.g., temperature)
        temperatures = experiment.config.get("temperature", [0.0])
        max_tokens = experiment.config.get("max_tokens", 512)
        
        for model_id in model_ids:
            for benchmark in benchmark_names:
                for temp in temperatures:
                    cfg = {"temperature": temp, "max_tokens": max_tokens}
                    # Normally this would dispatch asynchronously
                    self.run_benchmark(model_id, benchmark, cfg, experiment_id=experiment.id)
