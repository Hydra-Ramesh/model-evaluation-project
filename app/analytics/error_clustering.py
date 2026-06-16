import numpy as np
from typing import List, Dict, Any
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

class ErrorAnalyzer:
    """
    Analyzes model failures by clustering the questions they got wrong into semantic groups.
    Helps identify systemic blind spots (e.g., poor at logic puzzles, fails at Rust coding).
    """
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model_name)

    def cluster_failures(self, failed_questions: List[str], num_clusters: int = 4) -> Dict[str, Any]:
        """
        Embeds a list of failed questions and clusters them using KMeans.
        """
        if not failed_questions:
            return {"clusters": {}, "labels": []}
            
        # Encode failures into the dense semantic space
        embeddings = self.embedding_model.encode(failed_questions)
        
        # Ensure we don't request more clusters than we have data points
        actual_clusters = min(num_clusters, len(failed_questions))
        if actual_clusters == 0:
            return {"clusters": {}, "labels": []}
            
        # Run KMeans clustering
        kmeans = KMeans(n_clusters=actual_clusters, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(embeddings)
        
        # Group the raw string questions by their assigned cluster ID
        clusters: Dict[int, List[str]] = {i: [] for i in range(actual_clusters)}
        for i, label in enumerate(labels):
            clusters[label].append(failed_questions[i])
            
        # Format the output into a readable dictionary
        formatted_clusters = {}
        for cluster_id, questions in clusters.items():
            formatted_clusters[f"Cluster_{cluster_id}"] = questions
            
        return {
            "clusters": formatted_clusters,
            "labels": labels.tolist()
        }
