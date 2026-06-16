import os
import joblib
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sentence_transformers import SentenceTransformer

class LLMJudge:
    def __init__(self, model_path="data/llm_judge_model.pkl"):
        self.model_path = model_path
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.regressor = None
        
    def train(self, df: pd.DataFrame):
        print("Training LLM Judge (LightGBM)...")
        # Extract features
        texts = df['question'] + " [SEP] " + df['model_response']
        X_emb = self.embedder.encode(texts.tolist(), show_progress_bar=True)
        
        # Add numerical features
        X_num = df[['prompt_tokens', 'completion_tokens', 'semantic_score']].values
        
        import numpy as np
        X = np.hstack((X_emb, X_num))
        y = df['quality_score']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.regressor = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=6)
        self.regressor.fit(X_train, y_train)
        
        preds = self.regressor.predict(X_test)
        
        print(f"LLM Judge - MAE: {mean_absolute_error(y_test, preds):.4f}")
        print(f"LLM Judge - R2 Score: {r2_score(y_test, preds):.4f}")
        
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.regressor, self.model_path)
        print(f"Saved to {self.model_path}")
        
    def predict(self, question: str, response: str, prompt_tokens: int, completion_tokens: int, semantic_score: float) -> float:
        if self.regressor is None:
            self.regressor = joblib.load(self.model_path)
            
        import numpy as np
        emb = self.embedder.encode([question + " [SEP] " + response])
        num = np.array([[prompt_tokens, completion_tokens, semantic_score]])
        X = np.hstack((emb, num))
        
        score = self.regressor.predict(X)[0]
        # Clip to 1-10 range
        return float(np.clip(score, 1.0, 10.0))
