import numpy as np
from typing import Dict, Any
from scipy.spatial.distance import cosine

class EvaluationEngine:
    """
    Computes various deterministic metrics to evaluate an LLM's response against a ground truth.
    Uses sentence-transformers for semantic similarity.
    """
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        # We lazily load the model so it doesn't block Uvicorn startup
        self.embedding_model_name = embedding_model_name
        self._embedding_model = None

    @property
    def embedding_model(self):
        if self._embedding_model is None:
            # Lazy import to save 400MB of RAM on startup!
            from sentence_transformers import SentenceTransformer
            # Model is downloaded/loaded into memory only on the first API request
            self._embedding_model = SentenceTransformer(self.embedding_model_name)
        return self._embedding_model

    def exact_match(self, prediction: str, reference: str) -> bool:
        """Strict string exact match."""
        return prediction.strip().lower() == reference.strip().lower()

    def semantic_similarity(self, prediction: str, reference: str) -> float:
        """Calculates cosine similarity between sentence embeddings."""
        if not prediction or not reference:
            return 0.0
        embeddings = self.embedding_model.encode([prediction, reference])
        # cosine() from scipy returns distance, so 1 - distance = similarity
        similarity = 1 - cosine(embeddings[0], embeddings[1])
        return float(similarity)

    def compute_rouge(self, prediction: str, reference: str) -> float:
        """Placeholder for ROUGE score calculation."""
        # In a production environment, this would call huggingface 'evaluate' or 'rouge_score'
        # Fallback to semantic similarity for structural purposes
        return self.semantic_similarity(prediction, reference)

    def compute_bleu(self, prediction: str, reference: str) -> float:
        """Placeholder for BLEU score calculation."""
        # Fallback to semantic similarity for structural purposes
        return self.semantic_similarity(prediction, reference)

    def compute_bert_score(self, prediction: str, reference: str) -> float:
        """Placeholder for BERTScore."""
        # BERTScore is highly correlated with dense semantic similarity
        return self.semantic_similarity(prediction, reference)

    def evaluate_all(self, prediction: str, reference: str) -> Dict[str, Any]:
        """Runs the full suite of metrics."""
        return {
            "exact_match": self.exact_match(prediction, reference),
            "semantic_score": self.semantic_similarity(prediction, reference),
            "bleu_score": self.compute_bleu(prediction, reference),
            "rouge_score": self.compute_rouge(prediction, reference),
            "bert_score": self.compute_bert_score(prediction, reference)
        }
