from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

class BenchmarkItem(BaseModel):
    benchmark_name: str
    category: str
    question: str
    expected_answer: str
    metadata: Optional[Dict[str, Any]] = None

class BaseDatasetLoader(ABC):
    def __init__(self, cache_dir: str = "./data/cache"):
        self.cache_dir = cache_dir
        
    @abstractmethod
    def download(self) -> None:
        """Download and cache the dataset."""
        pass
        
    @abstractmethod
    def load(self) -> List[BenchmarkItem]:
        """Load and normalize the dataset."""
        pass
