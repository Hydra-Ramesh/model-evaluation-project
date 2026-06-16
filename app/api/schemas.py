from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class RunBenchmarkRequest(BaseModel):
    model_id: int
    benchmark_name: str
    config: Dict[str, Any] = {"temperature": 0.0, "max_tokens": 512}

class RunBenchmarkResponse(BaseModel):
    run_id: int
    status: str

class ExperimentCreate(BaseModel):
    name: str
    description: Optional[str] = None
    config: Dict[str, Any]
    model_ids: List[int]
    benchmark_names: List[str]

class ExperimentResponse(BaseModel):
    experiment_id: int
    message: str
