import pandas as pd
import numpy as np
import umap
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

class ErrorAnalyzer:
    def __init__(self):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.umap_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='cosine')
        self.kmeans = KMeans(n_clusters=5, random_state=42)
        
    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Takes the full dataframe, filters to failed examples, generates embeddings, 
        reduces with UMAP, and clusters with KMeans to automatically discover failure modes.
        """
        print("Running UMAP + KMeans Error Analysis...")
        failed_df = df[df['exact_match'] == False].copy()
        
        if failed_df.empty:
            print("No failures to analyze.")
            return failed_df
            
        texts = failed_df['question'] + " [SEP] " + failed_df['model_response']
        print("Generating embeddings for failures...")
        embeddings = self.embedder.encode(texts.tolist(), show_progress_bar=True)
        
        print("Reducing dimensions with UMAP...")
        umap_embeddings = self.umap_reducer.fit_transform(embeddings)
        
        print("Clustering with KMeans...")
        clusters = self.kmeans.fit_predict(umap_embeddings)
        
        failed_df['umap_x'] = umap_embeddings[:, 0]
        failed_df['umap_y'] = umap_embeddings[:, 1]
        failed_df['cluster'] = clusters
        
        # Auto-label heuristics based on known benchmarks/categories if present
        def label_cluster(c_id):
            cluster_data = failed_df[failed_df['cluster'] == c_id]
            # Dominant category in cluster
            # Our synthetic data has "error_category"
            if 'error_category' in cluster_data.columns and not cluster_data['error_category'].isnull().all():
                return cluster_data['error_category'].mode()[0]
            return f"Cluster {c_id}"
            
        cluster_names = {c: label_cluster(c) for c in range(5)}
        failed_df['cluster_label'] = failed_df['cluster'].map(cluster_names)
        
        print("Error Analysis Complete.")
        return failed_df
