import os
import joblib
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

class FailurePredictor:
    def __init__(self, model_path="data/failure_predictor.pkl"):
        self.model_path = model_path
        self.classifier = None
        
    def train(self, df: pd.DataFrame):
        print("Training Failure Predictor (CatBoost)...")
        # We predict if the model will FAIL (exact_match == 0) BEFORE execution
        # Thus, we cannot use model_response, completion_tokens, latency, or cost.
        features = ['model_name', 'benchmark_name', 'prompt_tokens', 'temperature', 'max_tokens']
        
        X = df[features].copy()
        # Convert categoricals
        X['model_name'] = X['model_name'].astype(str)
        X['benchmark_name'] = X['benchmark_name'].astype(str)
        
        y = (~df['exact_match']).astype(int) # 1 if failed, 0 if success
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.classifier = CatBoostClassifier(
            iterations=150, 
            learning_rate=0.1, 
            depth=6, 
            cat_features=['model_name', 'benchmark_name'],
            verbose=False
        )
        self.classifier.fit(X_train, y_train)
        
        preds = self.classifier.predict(X_test)
        
        print(f"Failure Predictor - Accuracy: {accuracy_score(y_test, preds):.4f}")
        print(f"Failure Predictor - F1 Score: {f1_score(y_test, preds):.4f}")
        
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.classifier, self.model_path)
        print(f"Saved to {self.model_path}")
