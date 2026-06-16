from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
from pydantic import BaseModel

class GenerationResult(BaseModel):
    response_text: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    cost: float

class BaseModelConnector(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 1024) -> GenerationResult:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: The input prompt string.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            
        Returns:
            A GenerationResult containing the response, latency, token usage, and cost.
        """
        pass
        
    @abstractmethod
    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """
        Calculate the cost of the generation based on token usage.
        """
        pass
