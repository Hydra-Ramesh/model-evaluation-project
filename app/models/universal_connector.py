import time
from typing import Dict, Any
import litellm
from .base import BaseModelConnector, GenerationResult

class UniversalModelConnector(BaseModelConnector):
    def __init__(self, model_name: str):
        """
        model_name should be formatted according to LiteLLM standard.
        Examples: 'gpt-4o', 'anthropic/claude-3-opus-20240229', 'deepseek/deepseek-chat', 'ollama/llama3'
        """
        super().__init__(model_name)

    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        # LiteLLM calculates cost dynamically based on model name
        try:
            cost = litellm.completion_cost(
                model=self.model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens
            )
            return cost
        except Exception:
            # Fallback if model cost is unmapped in LiteLLM yet
            return 0.0

    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 1024) -> GenerationResult:
        start_time = time.time()
        
        try:
            # litellm.completion abstracts over 100+ providers
            response = litellm.completion(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            latency_ms = (time.time() - start_time) * 1000.0
            response_text = response.choices[0].message.content or ""
            
            prompt_tokens = response.usage.prompt_tokens if response.usage else 0
            completion_tokens = response.usage.completion_tokens if response.usage else 0
            
            # Get exact cost automatically
            cost = self.calculate_cost(prompt_tokens, completion_tokens)
            
            return GenerationResult(
                response_text=response_text,
                latency_ms=latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost=cost
            )
            
        except Exception as e:
            # If an API key is missing or the connection fails, return a graceful error result
            # instead of crashing the entire experiment.
            return GenerationResult(
                response_text=f"[No Result] API Key missing or connection failed: {str(e)}",
                latency_ms=0.0,
                prompt_tokens=0,
                completion_tokens=0,
                cost=0.0
            )
